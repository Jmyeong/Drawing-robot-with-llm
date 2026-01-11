import os
import glob
import re
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

OVERFIT_MODE = True  
OVERFIT_SAMPLE_COUNT = 10

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_DIR = "./output_final" 
LOG_IMG_DIR = os.path.join(SAVE_DIR, "training_logs") 
TENSORBOARD_DIR = os.path.join(SAVE_DIR, "tensorboard_logs")
DATASET_DIR = "/ssd1/jm_data/project/Drawing_robot/Drawing-with-OMX/notebook/drawing/dataset_output_joint_data_2"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_IMG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

NUM_BINS = 256
MAX_STEPS = 300
SEQ_LEN = MAX_STEPS * 5 

BATCH_SIZE = 4       
LEARNING_RATE = 2e-4
EPOCHS = 100 if OVERFIT_MODE else 50 
VALIDATION_INTERVAL = 500 

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading LLM: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

action_tokens = [f"<bin_{i}>" for i in range(NUM_BINS)]
new_tokens = set(action_tokens) - set(tokenizer.vocab.keys())
tokenizer.add_tokens(list(new_tokens))
print(f"➕ Added {len(new_tokens)} action tokens.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    new_token_ids = tokenizer.convert_tokens_to_ids(list(new_tokens))
    input_embeddings[new_token_ids] = torch.nn.init.normal_(input_embeddings[new_token_ids], mean=0.0, std=0.02)
    output_embeddings[new_token_ids] = torch.nn.init.normal_(output_embeddings[new_token_ids], mean=0.0, std=0.02)

class RT2Dataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, num_bins=256, max_steps=100):
        self.tokenizer = tokenizer
        self.num_bins = num_bins
        self.max_steps = max_steps
        
        self.min_val = -np.pi
        self.max_val = np.pi
        
        search_pattern = os.path.join(dataset_dir, "*.npy")
        all_files = glob.glob(search_pattern)
        
        if len(all_files) == 0:
            raise ValueError(f"No .npy files found in {dataset_dir}")
        
        self.class_to_files = {} 
        self.file_paths = []

        temp_class_map = {}
        for p in all_files:
            filename = os.path.basename(p)
            category = filename.rsplit('_', 1)[0]
            
            if category not in temp_class_map: 
                temp_class_map[category] = []
            temp_class_map[category].append(p)

        if OVERFIT_MODE:
            print(f"⚠️ [OVERFIT MODE] Selecting 1 sample per class & Replicating {OVERFIT_SAMPLE_COUNT} times.")
            
            sorted_classes = sorted(temp_class_map.keys())
            for cls in sorted_classes:
                files = sorted(temp_class_map[cls])
                target_file = files[0] 
                
                self.file_paths.extend([target_file] * OVERFIT_SAMPLE_COUNT)
                
                self.class_to_files[cls] = [target_file]
                
                print(f"   Target Class: {cls:<15} -> {os.path.basename(target_file)} (x{OVERFIT_SAMPLE_COUNT})")
            
        else:
            self.file_paths = all_files
            self.class_to_files = temp_class_map
            print(f"✅ Found {len(self.file_paths)} trajectory files.")

    def float_to_bin(self, val, min_val, max_val):
        norm = (val - min_val) / (max_val - min_val)
        norm = np.clip(norm, 0.0, 1.0)
        return int(norm * (self.num_bins - 1))

    def __len__(self):
        return len(self.file_paths)
    
    def filter_and_smooth(self, traj, jump_threshold=0.3):
        cleaned_traj = traj.copy()
        seq_len, dims = cleaned_traj.shape
        
        for i in range(1, seq_len):
            for j in range(4): 
                diff = cleaned_traj[i, j] - cleaned_traj[i-1, j]
                if abs(diff) > jump_threshold:
                    cleaned_traj[i, j] = cleaned_traj[i-1, j]

        smoothed_traj = cleaned_traj.copy()
        window_size = 3
        kernel = np.ones(window_size) / window_size
        
        for j in range(4):
            smoothed = np.convolve(cleaned_traj[:, j], kernel, mode='same')
            smoothed[0] = cleaned_traj[0, j]
            smoothed[-1] = cleaned_traj[-1, j]
            smoothed_traj[:, j] = smoothed
            
        return smoothed_traj

    def resample_trajectory(self, traj, target_len):
        current_len = traj.shape[0]
        if current_len == target_len: return traj
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        new_traj = np.zeros((target_len, 5), dtype=np.float32)
        for i in range(5):
            new_traj[:, i] = np.interp(x_new, x_old, traj[:, i])
        return new_traj

    def __getitem__(self, idx):
        npy_path = self.file_paths[idx]
        
        filename = os.path.basename(npy_path)
        class_name = filename.rsplit('_', 1)[0]
        
        prompt = f"Draw a {class_name}"
        
        try:
            traj = np.load(npy_path).astype(np.float32)
        except:
            traj = np.zeros((self.max_steps, 5), dtype=np.float32)

        traj = self.filter_and_smooth(traj, jump_threshold=0.3)

        traj = self.resample_trajectory(traj, self.max_steps)
        
        token_strs = []
        for step in traj:
            for j in range(4):
                bin_idx = self.float_to_bin(step[j], self.min_val, self.max_val)
                token_strs.append(f"<bin_{bin_idx}>")
            pen_bin = self.float_to_bin(step[4], 0.0, 1.0)
            token_strs.append(f"<bin_{pen_bin}>")
        action_seq = "".join(token_strs)
        
        messages = [{"role": "system", "content": "You are a robot controller."}, {"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        full_text = prompt_text + action_seq + self.tokenizer.eos_token
        
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100 
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

class ValidationHandler:
    def __init__(self, model, tokenizer, save_dir, dataset, writer):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.dataset = dataset 
        self.min_val = dataset.min_val
        self.max_val = dataset.max_val
        self.writer = writer
        
    def decode_trajectory(self, token_string):
        bin_indices = [int(x) for x in re.findall(r"<bin_(\d+)>", token_string)]
        if not bin_indices: return None
        valid_len = (len(bin_indices) // 5) * 5
        if valid_len == 0: return None
        flat_data = np.array(bin_indices[:valid_len])
        traj = flat_data.reshape(-1, 5).astype(np.float32)
        
        for i in range(4):
            traj[:, i] = (traj[:, i] / 255.0) * (self.max_val - self.min_val) + self.min_val
        traj[:, 4] = traj[:, 4] / 255.0
        return traj

    def get_ground_truth(self, class_name):
        files = self.dataset.class_to_files.get(class_name, [])
        if files:
            target_path = files[0] 
            try:
                traj = np.load(target_path).astype(np.float32)
                traj = self.dataset.filter_and_smooth(traj, jump_threshold=0.3)
                return self.dataset.resample_trajectory(traj, self.dataset.max_steps)
            except: return None
        return None

    def run_validation(self, step_count):
        self.model.eval()
        
        available_classes = list(self.dataset.class_to_files.keys())
        if not available_classes: return

        random_cls = random.choice(available_classes)
        prompt = f"Draw a {random_cls}"
        
        print(f"🔎 Validating: {prompt}")

        cls_name_extract = prompt.replace("Draw a ", "").strip()
        gt_traj = self.get_ground_truth(cls_name_extract)

        messages = [{"role": "system", "content": "You are a robot controller."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=600, 
                min_new_tokens=50, 
                do_sample=False, 
                repetition_penalty=1.0, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        gen_traj = self.decode_trajectory(gen_text)
        
        if gen_traj is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = ["J1", "J2", "J3", "J4", "Pen"]
            colors = ['r', 'g', 'b', 'c', 'k']
            
            for i in range(5):
                plt.plot(gen_traj[:, i], label=f"Gen {labels[i]}", color=colors[i], linestyle='-', alpha=0.8)
            
            if gt_traj is not None:
                x_axis = np.linspace(0, len(gen_traj)-1, len(gt_traj))
                for i in range(5):
                    plt.plot(x_axis, gt_traj[:, i], label=f"GT {labels[i]}", color=colors[i], linestyle='--', alpha=0.5, linewidth=2)
            
            ax.set_title(f"Step {step_count}: {prompt}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            ax.grid(True)
            
            self.writer.add_figure(
                f"Val/Trajectory_{cls_name_extract}",
                fig,
                global_step=step_count
            )
            plt.close(fig)
            
        self.model.train()

if __name__ == "__main__":
    ds = RT2Dataset(DATASET_DIR, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    writer = SummaryWriter(TENSORBOARD_DIR)
    val_handler = ValidationHandler(model, tokenizer, LOG_IMG_DIR, dataset=ds, writer=writer)
    
    print(f"🔥 Start Training (Overfit: {OVERFIT_MODE}, Epochs: {EPOCHS})")
    model.train()
    
    global_step = 0
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            if global_step % 100 == 0:
                writer.add_scalar("Training/Step_Loss", loss.item(), global_step)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if global_step % VALIDATION_INTERVAL == 0:
                val_handler.run_validation(global_step)

        avg_epoch_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")
        
        writer.add_scalar("Training/Epoch_Loss", avg_epoch_loss, epoch + 1)
        
        save_path = os.path.join(SAVE_DIR, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    writer.close()
    print("🎉 Training Complete!")