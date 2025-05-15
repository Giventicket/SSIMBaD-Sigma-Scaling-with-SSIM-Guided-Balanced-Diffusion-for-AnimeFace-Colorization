# AnimeDiffusion - PyTorch Lightning Implementation

This project is a PyTorch Lightning-based implementation of the original AnimeDiffusion repository ([xq-meng/AnimeDiffusion](https://github.com/xq-meng/AnimeDiffusion)). The code has been restructured and adapted to take advantage of PyTorch Lightning's modularity, scalability, and ease of experimentation. This README explains the purpose of each file and its role in the training and testing pipeline.

---

## **Overview**
AnimeDiffusion is a diffusion-based generative model designed to generate high-quality anime-style images. The implementation here includes key improvements and optimizations for distributed training, logging, and configuration management using PyTorch Lightning.

### **Features**
- Easy configuration via `argparse`
- Distributed training support (`DDP` strategy)
- TensorBoard integration for training visualization
- Flexible checkpointing and model saving
- Reproducible training runs with seed setting

---

## **Key Files**

### **`AnimeDiffusion.py`**
This file contains the core implementation of the `AnimeDiffusion` model. The model integrates:
- UNet architecture with customizable channels and attention heads.
- A diffusion process with configurable time steps and beta schedules.
- Optional CBAM (Convolutional Block Attention Module) for enhanced feature extraction.

### **`main.py`**
The entry point of the project, responsible for:
1. **Argument Parsing**: Configures all training and testing parameters using `argparse`.
2. **Trainer Setup**: Initializes PyTorch Lightning's `Trainer` with distributed training support and logging.
3. **Callbacks**: Includes a `ModelCheckpoint` callback to save the top-3 models based on training loss.
4. **Training and Testing**: Handles the execution of the training and testing loops.

---

## **How to Run**

### **Install Dependencies**
Ensure you have the necessary Python packages installed. You can create a `requirements.txt` from the extracted dependencies and install them:

```bash
pip install -r requirements.txt
```

### **Run Training**
To train the model, run:
```bash
python main.py --do_train True --do_test False --epochs 50
```

### **Run Testing**
To test the model using a specific checkpoint, run:
```bash
python main.py --do_train False --do_test True --test_output_dir ./result/
```

### **Custom Configurations**
You can override the default configurations by passing arguments to the script. For example:
```bash
python main.py --lr 5e-5 --train_batch_size 16 --gpus 0 1
```

---

## **Arguments**

Below are the key arguments you can configure in `main.py`:

### **General**
- `--do_train`: Enable or disable training (default: `True`).
- `--do_test`: Enable or disable testing (default: `True`).
- `--epochs`: Number of training epochs (default: `50`).
- `--gpus`: List of GPUs to use (default: `[0, 1]`).

### **Diffusion Process**
- `--time_step`: Number of diffusion time steps (default: `1000`).
- `--linear_start`, `--linear_end`: Start and end values for the linear beta schedule.

### **UNet Model**
- `--channel_in`, `--channel_out`: Input and output channels (default: `7` and `3`).
- `--channel_mult`: List of channel multipliers (default: `[1, 2, 4, 8]`).
- `--attention_head`: Number of attention heads in the UNet.
- `--cbam`: Whether to enable CBAM (default: `False`).

### **Data Paths**
- `--train_reference_path`: Path to reference training data.
- `--train_condition_path`: Path to conditional training data.
- `--test_reference_path`: Path to reference testing data.
- `--test_condition_path`: Path to conditional testing data.

### **Optimizer**
- `--lr`: Initial learning rate (default: `1e-4`).
- `--weight_decay`: Weight decay for optimizer (default: `0.01`).
- `--warmup_epochs`: Number of warmup epochs (default: `1`).

---

## **Logs and Checkpoints**

- **TensorBoard**: Logs are stored in the `logs/` directory. You can visualize training progress with:
  ```bash
  tensorboard --logdir logs
  ```
- **Checkpoints**: Saved in `logs/lightning_logs/version_0/checkpoints/`. The best 3 checkpoints are saved based on training loss.

---

## **Attribution**

This project is based on [xq-meng/AnimeDiffusion](https://github.com/xq-meng/AnimeDiffusion). The original code was restructured and ported to PyTorch Lightning for easier experimentation and distributed training.

For more information, please refer to the original repository or raise issues here for discussion.