# kontext-vtoff

If on a server, you may need to install `conda`, or alternatively `miniconda`:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Create a new Conda environment:
```bash
conda create -n kontext python=3.11
conda activate kontext
```
Then, clone this repository:
```bash
git clone https://github.com/rizavelioglu/kontext-vtoff.git
cd kontext-vtoff
pip install -r requirements.txt
```

Login to your HuggingFace account via cli, which is necessary to download FLUX model:
```bash
hf auth login
```

Start training with a single-GPU, i.e. H100:
```bash
python train.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.1-Kontext-dev  \
  --vae_encode_mode="mode" \
  --output_dir="./flux-kontext-lora/" \
  --dataset_name="./data/training/" \
  --image_column="end" \
  --cond_image_column="start" \
  --caption_column="instruction" \
  --repeats=1 \
  --max_sequence_length=512 \
  --num_validation_images=4 \
  --validation_epochs=50 \
  --rank=16 \
  --lora_alpha=4 \
  --lora_dropout=0.0 \
  --seed=42 \
  --resolution=1024 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=200 \
  --dataloader_num_workers=0 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-4 \
  --adam_epsilon=1e-8 \
  --logging_dir="logs" \
  --allow_tf32 \
  --mixed_precision="bf16"
```

Run inference with:
```bash
python inference.py --input_folder "./data/testing/" --output_folder "./output/" --lora_folder "./flux-kontext-lora"
```

## Helpful Links
- [inference pipeline (single-image)][2]
- [inference pipeline (multi-image)][3]
- [HF-space (multi-image)][4]
- [I2I training script][5]
- [inference pipeline (inpaint)][6]

<!-- References -->
[1]: https://arxiv.org/abs/2506.15742
[2]: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
[3]: https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_flux_kontext_multiple_images.py
[4]: https://huggingface.co/spaces/kontext-community/FLUX.1-Kontext-multi-image
[5]: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux_kontext.py
[6]: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext_inpaint.py
