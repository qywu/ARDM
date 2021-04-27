# ARDM
Alternate Recurrent Dialog Model

I have written two colab files. Hope it will help you.

Training: https://colab.research.google.com/drive/1RmC3-kyNdSEKrzUVcPB3lRmn_lnCi66T

Inference: https://colab.research.google.com/drive/1ib7YCeNhkIDAzuOKotSlw1CfIBP_zE4r

## Citation

You can cite the paper with:

```
@inproceedings{wu-etal-2021-alternating,
    title = "Alternating Recurrent Dialog Model with Large-scale Pre-trained Language Models",
    author = "Wu, Qingyang  and
      Zhang, Yichi  and
      Li, Yu  and
      Yu, Zhou",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.110",
    pages = "1292--1301",
    abstract = "Existing dialog system models require extensive human annotations and are difficult to generalize to different tasks. The recent success of large pre-trained language models such as BERT and GPT-2 (Devlin et al., 2019; Radford et al., 2019) have suggested the effectiveness of incorporating language priors in down-stream NLP tasks. However, how much pre-trained language models can help dialog response generation is still under exploration. In this paper, we propose a simple, general, and effective framework: Alternating Recurrent Dialog Model (ARDM). ARDM models each speaker separately and takes advantage of the large pre-trained language model. It requires no supervision from human annotations such as belief states or dialog acts to achieve effective conversations. ARDM outperforms or is on par with state-of-the-art methods on two popular task-oriented dialog datasets: CamRest676 and MultiWOZ. Moreover, we can generalize ARDM to more challenging, non-collaborative tasks such as persuasion. In persuasion tasks, ARDM is capable of generating human-like responses to persuade people to donate to a charity.",
}
```
