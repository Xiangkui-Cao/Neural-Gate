## Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating (Accepted by ECCV 2026)

Neural Gate is a neuron-level model editing framework for privacy protection in large vision-language models (LVLMs). The project focuses on editing privacy-sensitive concepts so that a model refuses unsafe/private queries while preserving normal visual-language capabilities on unrelated questions.
> **Paper link:** https://arxiv.org/abs/2603.12598.  
> **Paper abstract:** Large Vision-Language Models (LVLMs) have shown remarkable potential across a wide array of vision-language tasks, leading to their adoption in critical domains such as finance and healthcare. However, their growing deployment also introduces significant security and privacy risks. Malicious actors could potentially exploit these models to extract sensitive information, highlighting a critical vulnerability. Recent studies show that LVLMs often fail to consistently refuse instructions designed to compromise user privacy. While existing work on privacy protection has made meaningful progress in preventing the leakage of sensitive data, they are constrained by limitations in both generalization and non-destructiveness. They often struggle to robustly handle
unseen privacy-related queries and may inadvertently degrade a model's performance on standard tasks. To address these challenges, we introduce Neural Gate, a novel method for mitigating privacy risks through neuron-level model editing. Our method improves a model's privacy safeguards by increasing its rate of refusal for privacy-related questions, crucially extending this protective behavior to novel sensitive queries not encountered during the editing process. Neural Gate operates by learning a feature vector to identify neurons associated with privacy-related concepts within the model's representation of a subject. This localization then precisely guides the update of model parameters. Through comprehensive experiments on MiniGPT and LLaVA, we demonstrate that our method significantly boosts the model's privacy protection while preserving its original utility.

## Repository Structure
```text
.
├── data/                  # Neural Gate / PrivacyPair-style editing datasets in JSON format
├── dsets/                 # Dataset loaders and benchmark wrappers
├── experiments/           # Main experiment, evaluation, layer-localization, and summary scripts
├── hparams/Ours/          # Hyperparameters for Neural Gate on LLaVA and MiniGPT-4
├── llava/                 # LLaVA model/evaluation code used by experiments
├── minigpt4/              # MiniGPT-4 model/evaluation code used by experiments
├── memit/                 # MEMIT baseline/editing utilities
├── model_func/            # Model helper functions
├── ours/                  # Neural Gate editing, mask construction, and weight update logic
├── rome/                  # ROME baseline/editing utilities
├── tool/                  # Mask and function-dictionary utilities
├── util/                  # Shared helpers, global paths, generation, hooks, and stats
├── globals.yml            # Project paths for data, results, hparams, and image root
├── requirements.txt       # Python dependency list
└── results.py             # Result processing helper
```
## Main Experiments
Most experiments should be launched from the `experiments/` folder or by calling scripts under `experiments/` from the repository root.
Script	Purpose
`experiments/evaluate_llava.py`	Run Neural Gate and baseline editing/evaluation on LLaVA-1.5-7B.
`experiments/evaluate_minigpt.py`	Run Neural Gate and baseline editing/evaluation on MiniGPT4-LLaMA2-7B.
`experiments/layer_loc_llava.py`	Layer/neuron localization analysis for LLaVA.
`experiments/layer_loc_minigpt.py`	Layer/neuron localization analysis for MiniGPT-4.
## Datasets
The repository includes Neural Gate privacy editing dataset JSON files under `data/`:
`data/neural_gate_documents_en.json`
`data/neural_gate_military_vehicles.json`
`data/neural_gate_Passport.json`
`data/neural_gate_Phone_no.json`
`data/neural_gate_Receipts.json`
`data/neural_gate_Student_ID.json`
Each sample contains an image path, a `requested_rewrite` target, paraphrase prompts, neighborhood prompts, and generation prompts. These files correspond to the PrivacyPair-style setting described in the paper: sensitive prompts test privacy protection, while neighboring/benign prompts test whether normal non-sensitive behavior is preserved.
The paper studies six privacy-sensitive subjects: phone numbers, student IDs, receipts, passports, military equipment, and government documents. Configure the image root with `IMAGE_ROOT` in `globals.yml`; image paths in the JSON files are resolved relative to that root.
## Setup
Create an environment and install dependencies:
```bash
conda create -n neural-gate python=3.9
conda activate neural-gate
pip install -r requirements.txt
```
Then update paths in `globals.yml` (Images are available at https://github.com/Xiangkui-Cao/Multi-P2A):
```yaml
RESULTS_DIR: "results"
DATA_DIR: "data"
STATS_DIR: "data/stats"
HPARAMS_DIR: "hparams"
IMAGE_ROOT: "/path/to/privacy/images"
```
You also need local checkpoints for the target LVLMs:
LLaVA-1.5-7B for `experiments/evaluate_llava.py`;
MiniGPT-4 LLaMA2-7B and its eval config for `experiments/evaluate_minigpt.py`.
> Note: some scripts currently contain machine-specific checkpoint paths and CUDA settings. Before running, replace those paths with your local model paths and set `CUDA_VISIBLE_DEVICES` / `--gpu-id` according to your hardware.
## Hyperparameters
Neural Gate hyperparameters are stored in `hparams/Ours/`:
`hparams/Ours/llava1.5-7b.json`: default LLaVA setting, editing `model.layers.{}.mlp.down_proj`.
`hparams/Ours/minigpt4_llama2_7b.json`: default MiniGPT-4 setting, editing `llama_model.model.layers.{}.mlp.down_proj`.
Important fields include:
`layers`: edited transformer layer IDs;
`num_steps`: optimization steps for editing;
`lr`: editing learning rate;
`kl_factor`: locality/regularization weight;
`rewrite_module_tmp`: module weight template to edit;
`layer_module_tmp`: transformer layer template used for activation/mask computation.
Running Neural Gate
LLaVA-1.5
Example command:
```bash
python experiments/evaluate_llava.py \
  --alg_name Ours \
  --model_name llava1.5-7b \
  --hparams_fname llava1.5-7b.json 
```
By default, the script evaluates Neural Gate on configured LLaVA layers. Edit the layer loop near the bottom of `experiments/evaluate_llava.py` or pass/modify `layer_check` in code for custom layer localization runs.
MiniGPT-4
Example command:
```bash
python experiments/evaluate_minigpt.py \
  --alg_name Ours \
  --model_name minigpt4_llama2_7b \
  --cfg-path /path/to/minigpt4_llama2_eval.yaml \
  --hparams_fname minigpt4_llama2_7b.json 
```
The MiniGPT-4 script also contains default loops over Neural Gate layer settings near the bottom of `experiments/evaluate_minigpt.py`.


## Citation
If you use Neural Gate, please cite the paper once the official citation is available:
```bibtex
@article{cao2026neural,
  title={Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating},
  author={Cao, Xiangkui and Zhang, Jie and Kan, Meina and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2603.12598},
  year={2026}
}
```
## Notes
The main reproducibility path is through `experiments/`, not through standalone package entry points.
`globals.yml` controls project-level paths used by loaders and evaluators.
Large model checkpoints and image assets are not included in this repository.
The included PDF can be used to complete the paper link, figure, method description, and citation details when the final version is available.
