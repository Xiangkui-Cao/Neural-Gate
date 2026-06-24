from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
import torch
from model_func.model_func import model_outputs

def generate_fast(model, tok, imgs, prompts, max_out_len: int = 200):
    max_new_tokens = max_out_len
    min_length = 1
    max_length = 2000

    num_beams = 1
    top_p = 0.9
    repetition_penalty = 1.05
    length_penalty = 1
    temperature = 1.0
    # imgs = model.model.encode_img(torch.zeros(1, 3, 224, 224).to(model.device))[0][0].unsqueeze(0)

    outs = []
    for prompt in prompts:
        conv = model.get_conv([prompt], [""])[0]
        embs, _ = model.get_context_emb(conv, imgs)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:, :]
        if model.model_name == "llava1.5-7b":
            input_ids = tokenizer_image_token(
                conv.get_prompt(),
                model.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).cuda()
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"imgs shape: {imgs[0].shape}")
            # print(f"Vision tower: {model.model.vision_tower}")
            generation_dict = dict(
                inputs_embeds=embs,
                # inputs=input_ids,
                # images=imgs[0],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                use_cache=True,
                temperature=float(temperature),
                repetition_penalty=repetition_penalty,
            )
        else:
            generation_dict = dict(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=float(temperature),
            )
        print("-"*20)
        print(f"USER: {prompt}")
        with torch.no_grad():
            if model.model_name == "llava1.5-7b":
                # output_token = model.model.model(inputs_embeds)
                # logits = model_outputs(model, inputs_embeds=embs,
                #                        attention_mask=torch.ones(embs.shape[:-1], device=embs.device)).logits  # torch.Size([1, 321, 32000])
                # print(f"logits shape: {logits.shape}")
                # log_probs = torch.log_softmax(logits, dim=2)
                # print(f"log_probs: {log_probs}")
                # test_log = log_probs[:, -20:, :]
                # test_log = torch.argmax(test_log, dim=2, keepdim=False)
                # print(f"test_log: {tok.batch_decode(test_log, skip_special_tokens=False)}")
                output_token = model.model.generate(**generation_dict)[0]
                output_text = model.tokenizer.decode(output_token, skip_special_tokens=True)
            else:
                output_token = model.model.llama_model.generate(**generation_dict)[0]
                output_text = model.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        # print(f"Generating prefix: {output_text}")
        outs.append(output_text)
        print(f"MODEL: {output_text}")
        # outs.append("")
    return outs