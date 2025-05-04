# Pragmatic Radiology Report Generation: Exploring Reproducibility & Extensions

By: Ethan Rasmussen (ethanmr3@illinois.edu)

To read the project paper, **[CLICK HERE](https://github.com/ethanrasmussen/llm_radiology/blob/main/CS598-DL4H-FinalPaper.pdf)**

To watch the presentation, **[CLICK HERE]**

This code mainly works for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), but can be adapted slightly to work with other chest X-ray datasets.

## The original paper introduces the following:

* Report cleaning with large language models
* Training an image classifier for detecting positive findings
* Finetuning LLaMA-2-7B on predicted conditions and indications
* Generating reports with Pragmatic-LLaMA
* Evaluating generated reports

## This repository follows the reproduction of the original paper, as well as introducing the following extensions:

* Finetuning with LLaMA-3.1-8B-Instruct, a newer language model, which appears to significantly reduce hallucination as measured by heuristic
* Experimentation with chain-of-thought and structured prompting methods for enhanced inference results
* *Impression pruning*, a data post-processing step which cleans up noisy model inference outputs
* *NOTE: Reproduction was done with a subset of MIMIC-CXR, due to computational resource constraints*

## Run existing code from checkpoints for inference:
NOTE: The full pipeline notebook, at [./dl4h_notebooks/LLM_Radiology_Pipeline.ipynb](https://github.com/ethanrasmussen/llm_radiology/blob/main/dl4h_notebooks/LLM_Radiology_Pipeline.ipynb), provides follow-along code for all steps except training/fine-tuning. It may require `git clone` or download to view, as its size/format make it non-viewable in GitHub's web interface.

1. Install the necessary dependencies via `pip install -q -r requirements.txt` and `pip install -q -r eval_requirements.txt`
2. Acquire MIMIC-CXR (or MIMIC-CXR-JPG) dataset (or subset), and organize the file structure such that a tuple of (image, indication) can be fed into the model. For additional guidance, view the data download instructions in the [paper.](https://github.com/ethanrasmussen/llm_radiology/blob/main/CS598-DL4H-FinalPaper.pdf)
3. If you'd like to reproduce the results by training a model, skip to the section below. To run a model from checkpoints, you'll simply need to load in a model from Huggingface. More info is available in [./dl4h/z2_prep_models.py](https://github.com/ethanrasmussen/llm_radiology/blob/main/dl4h/z2_prep_models.py)
4. Finally, run the pragmatic inference script below. To utilize different prompting styles, change the value of `--instruct_path`. Possible arguments include: `instructions.txt` (base), `CoT_instructions.txt` (chain-of-thought), and `structured_instructions.txt`

```
python pragmatic_llama_inference.py --llama_path <insert> --indication_path <insert> --vision_path <insert> --image_path <insert> --vision_out_path <insert> --outpath <insert> --instruct_path ./prompts/report_writing/<insert>
```


## Run code for training/fine-tuning:
NOTE: The finetune notebook, at [./dl4h_notebooks/LLM_Radiology_Finetune.ipynb](https://github.com/ethanrasmussen/llm_radiology/blob/main/dl4h_notebooks/LLM_Radiology_Finetune.ipynb), provides follow-along code for training/fine-tuning a LLaMA model using the pragmatic approach.

1. Install the necessary dependencies via `pip install -q -r requirements.txt` and `pip install -q -r eval_requirements.txt`
2. Acquire MIMIC-CXR (or MIMIC-CXR-JPG) dataset (or subset), and organize the file structure such that a tuple of (image, indication) can be fed into the model. For additional guidance, view the data download instructions in the [paper.](https://github.com/ethanrasmussen/llm_radiology/blob/main/CS598-DL4H-FinalPaper.pdf)
3. Clean your radiology reports, via `deepspeed --num_gpus=<insert> report_cleaning.py --chexbert_path <insert> --dataset_path <insert> --output_dir <insert>`
4. Format your training data via `python format_llama_input.py --indication_path <insert> --impression_path <insert> --outpath <insert>`
5. Run the training script via `accelerate launch finetune_qLoRA.py`, with all arguments included


## Run code for evaluation of results:
NOTE: The full pipeline notebook, at [./dl4h_notebooks/LLM_Radiology_Pipeline.ipynb](https://github.com/ethanrasmussen/llm_radiology/blob/main/dl4h_notebooks/LLM_Radiology_Pipeline.ipynb), provides follow-along code for all steps except training/fine-tuning. It may require `git clone` or download to view, as its size/format make it non-viewable in GitHub's web interface.

1. Follow the steps from above sections to ensure your model results have been generated.
2. Run `python evaluate.py --gt_path <insert> --gen_path <insert> --out_path <insert>`, where `gt_path` is the filepath to ground truth data, `gen_path` is the filepath to generated reports, and `out_path` is the desired location for evaluation output.

Note that much of the evaluation logic draws from [Yu, Endo, and Krishnan et al.](https://github.com/rajpurkarlab/CXR-Report-Metric/blob/main/CXRMetric/run_eval.py) [11], located in the file `CXRMetric/run_eval.py`



## Generating reports with Pragmatic-LLaMA

Insert the path to your finetuned Pragmatic-LLaMA model, path to indications, path to the directory containing the vision model and tuned classification thresholds, and specify an output path for predicted vision labels. This helps save time on subsequent runs on the same images by not having to re-run the classifier.



## References:

[1] Nguyen, D., Chen, C., He, H., & Tan, C. (2023). **Pragmatic Radiology Report Generation**. *Proceedings of Machine Learning Research (ML4H)*, 225, 1–16. 2

[2] Sonoda, Y. et al. (2024). **Structured clinical reasoning prompt enhances LLM’s diagnostic capabilities in diagnosis please quiz cases**. *Japanese Journal of Radiology*, 43:586–592 3

[3] Wang, G. et al. (2023). **ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data**. *arXiv*, 2306.09968v1 4

[4] Liu, Z. et al. (2023). **Radiology Llama2: Best in Class Large Language Model for Radiology**. *arXiv*, 2309.06419v1 5

[5] Wang, L., Chen, X., Deng, X. et al. (2024). **Prompt engineering in consistency and reliability with the evidence-based guideline for LLMs**. *npj Digital Medicine*. 7, 416

[6] Jain, S. et al. (2021). **RadGraph: Extracting Clinical Entities and Relations from Radiology Reports**. *arXiv*, 2106.14463 7

[7] Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q. Artzi, Y. (2019). **BERTScore: Evaluating Text Generation with BERT**. *arXiv*, 1904.09675 8

[8] Papineni, K., Roukos, S., Ward, T. Zhu, W. (2002). **BLEU: a Method for Automatic Evaluation of Machine Translation**. *Proceedings of the Association for Computational Linguistics (ACL)*, 40:311–318 9

[9] Smit, A. et al. (2020). **CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT**. *arXiv*, 2004.0916710

[10] Johnson, A.E.W., Pollard, T.J., Berkowitz, S.J. et al. (2019). **MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports**. *Nature: Scientific Data*. 6, 317

[11] Yu, Feiyang, Mark Endo, Rayan Krishnan, Ian Pan, Andy Tsai, Eduardo Pontes Reis, Eduardo Kaiser Ururahy Nunes Fonseca et al. (2023). **Evaluating progress in automatic chest x-ray radiology report generation**. *Patterns*. 4, no. 9