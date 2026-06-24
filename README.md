Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating
Neural Gate is a neuron-level model editing framework for privacy protection in large vision-language models (LVLMs). The project focuses on editing privacy-sensitive visual concepts so that a model refuses or redirects unsafe/private queries while preserving normal visual-language capabilities on unrelated questions.
> **Paper link:** TODO: add arXiv / conference paper URL here.  
> **Paper figure:** TODO: add the main method/framework figure here, for example `assets/neural_gate_overview.png`.
Official code for Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating.
Overview
Neural Gate studies privacy-oriented editing for LVLMs such as LLaVA-1.5-7B and MiniGPT4-LLaMA2-7B. The paper targets a model-compliance privacy risk: an LVLM may reveal sensitive information from an input image when asked by a malicious or privacy-invasive instruction, even if that information was not memorized from training data.
Given paired image-question samples containing the same privacy subject, Neural Gate identifies neurons that consistently encode privacy-related concepts and uses a local gradient truncation mechanism to edit model parameters. The edited model is evaluated on whether it:
blocks or changes answers for privacy-sensitive rewrite prompts;
generalizes to paraphrased sensitive prompts;
preserves behavior on neighboring non-sensitive prompts;
maintains utility on general LVLM benchmarks.
The main implementation is in `ours/`, while experiment entry points and evaluation scripts are in `experiments/`.
Paper Highlights
Neuron-level gradient gating: Neural Gate learns feature/gating vectors to localize privacy-related neurons and truncates gradients from inactive or weakly activated neurons during editing.
PrivacyPair dataset: The paper constructs paired sensitive/benign samples that share the same privacy subject but differ in privacy sensitivity, encouraging the model to distinguish privacy intent from normal semantics.
Generalized privacy protection: The method aims to refuse unseen privacy-related queries by capturing privacy concepts rather than memorizing training prompts or keywords.
Non-destructive editing: By restricting updates to privacy-relevant neurons, Neural Gate reduces unwanted degradation on benign questions and standard LVLM tasks.
Evaluation on two LVLMs: Experiments are conducted on MiniGPT4-LLaMA2-7B and LLaVA-1.5-7B, with safety and utility measured on PrivacyPair-test, MLLMGuard, ScienceQA, MME, and POPE.
Repository Structure
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
Main Experiments
Most experiments should be launched from the `experiments/` folder or by calling scripts under `experiments/` from the repository root.
Script	Purpose
`experiments/evaluate_llava.py`	Run Neural Gate and baseline editing/evaluation on LLaVA-1.5-7B.
`experiments/evaluate_minigpt.py`	Run Neural Gate and baseline editing/evaluation on MiniGPT4-LLaMA2-7B.
`experiments/layer_loc_llava.py`	Layer/neuron localization analysis for LLaVA.
`experiments/layer_loc_minigpt.py`	Layer/neuron localization analysis for MiniGPT-4.
`experiments/causal_trace.py`	Causal tracing utilities for locating influential components.
`experiments/sweep.py`	Hyperparameter/layer sweep utilities.
`experiments/summarize.py`	Aggregate run outputs saved under `results/`.
`experiments/test_pt.py`	MiniGPT-4 probing/testing helper.
`experiments/test_pt_llava.py`	LLaVA probing/testing helper.
Datasets
The repository includes Neural Gate privacy editing dataset JSON files under `data/`:
`data/neural_gate_documents_en.json`
`data/neural_gate_military_vehicles.json`
`data/neural_gate_Passport.json`
`data/neural_gate_Phone_no.json`
`data/neural_gate_Receipts.json`
`data/neural_gate_Student_ID.json`
Each sample contains an image path, a `requested_rewrite` target, paraphrase prompts, neighborhood prompts, and generation prompts. These files correspond to the PrivacyPair-style setting described in the paper: sensitive prompts test privacy protection, while neighboring/benign prompts test whether normal non-sensitive behavior is preserved.
The paper studies six privacy-sensitive subjects: phone numbers, student IDs, receipts, passports, military equipment, and government documents. Configure the image root with `IMAGE_ROOT` in `globals.yml`; image paths in the JSON files are resolved relative to that root.
Setup
Create an environment and install dependencies:
```bash
conda create -n neural-gate python=3.10
conda activate neural-gate
pip install -r requirements.txt
```
Then update paths in `globals.yml`:
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
Hyperparameters
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
  --hparams_fname llava1.5-7b.json \
  --ds_name mcf \
  --dataset_size_limit 10 \
  --num_edits 3 \
  --questions_per_img 5 \
  --privacy_value 0
```
By default, the script evaluates Neural Gate on configured LLaVA layers. Edit the layer loop near the bottom of `experiments/evaluate_llava.py` or pass/modify `layer_check` in code for custom layer localization runs.
MiniGPT-4
Example command:
```bash
python experiments/evaluate_minigpt.py \
  --alg_name Ours \
  --model_name minigpt4_llama2_7b \
  --cfg-path /path/to/minigpt4_llama2_eval.yaml \
  --hparams_fname minigpt4_llama2_7b.json \
  --ds_name mcf \
  --dataset_size_limit 10 \
  --num_edits 3 \
  --questions_per_img 5 \
  --privacy_value 0 \
  --gpu-id 0
```
The MiniGPT-4 script also contains default loops over Neural Gate layer settings near the bottom of `experiments/evaluate_minigpt.py`.
Evaluation and Summaries
Experiment outputs are saved under `results/<alg_name>/...` by default. To summarize a run directory:
```bash
python experiments/summarize.py --dir_name Ours
```
To summarize selected runs only:
```bash
python experiments/summarize.py --dir_name Ours --runs run_000,run_001
```
The summary script reports editing efficacy, paraphrase generalization, neighborhood specificity, and aggregate scores when the corresponding result fields are available.
In the paper, the main metrics are:
RtA / Refusal Rate: refusal behavior on privacy-sensitive samples.
EtA: average of refusal rate on sensitive samples and non-refusal behavior on insensitive samples, used as a joint privacy-utility measure on PrivacyPair-test.
ACC: standard accuracy for utility benchmarks such as ScienceQA, MME, and POPE.
The paper reports PrivacyPair-test and MLLMGuard as safety evaluations, and ScienceQA, MME, and POPE as utility evaluations.
Baselines
The evaluation scripts include hooks for several editing baselines, including MEMIT, ROME, FT, MEND, DINM, and AlphaEdit. The paper compares against representative editing/unlearning approaches such as MEMIT, AlphaEdit, DINM, SKU, and MemFlex. Some baseline imports may require additional code or dependencies that are not included in this repository snapshot. Neural Gate-specific code is available under `ours/` and uses the `Ours` algorithm entry in the experiment scripts.
Paper Assets
Please fill in these items after the paper page/assets are finalized:
```markdown
[Paper](TODO)

![Neural Gate overview](TODO: path/to/figure.png)
```
Recommended figure placement:
```text
assets/
└── neural_gate_overview.png
```
Citation
If you use Neural Gate, please cite the paper once the official citation is available:
```bibtex
@inproceedings{TODO_neural_gate,
  title     = {Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating},
  author    = {Xiangkui Cao and Jie Zhang and Meina Kan and Shiguang Shan and Xilin Chen},
  booktitle = {TODO},
  year      = {2026}
}
```
Notes
The main reproducibility path is through `experiments/`, not through standalone package entry points.
`globals.yml` controls project-level paths used by loaders and evaluators.
Large model checkpoints and image assets are not included in this repository.
The included PDF can be used to complete the paper link, figure, method description, and citation details when the final version is available.