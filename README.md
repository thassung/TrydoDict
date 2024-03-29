# TrydoDict

## Overview

   Welcome to the Machine Translation Demo App! This web-based application demonstrates the basic functionality of a machine translation, allowing users to input a word or sentence in English and retrieve the generated translation in Thai.

## Dataset

   The machine translation is trained with a data from [Hugging Face](https://huggingface.co/datasets/scb_mt_enth_2020) which is a large English-Thai parallel corpus, curated from news, Wikipedia articles, SMS messages, task-based dialogs, web-crawled data and government documents.

   There are 801,402 rows of English-Thai parallel text in the train dataset, 100,173 rows in the validation dataset, and 100,177 rows in the test dataset. In this repository, the dataset is truncated to 10,000 rows of train dataset, 2,000 rows in validation dataset, and 2,000 rows in test dataset.

## Features

   - **Input:** User can enter an English word or sentence in the provided input section.
   - **Submit Button:** User clicks *submit* after typing the prompt. The app will generate the translation from the provided text.
   - **Generated Translation:** A translation will be generated and shown next to the submit button. The app might give &lt;unk&gt; in the generated text if the input contains unknown word or when the app sees that is appropriate.

## Application

### Prerequisites

- Ensure you have Python installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thassung/TrydoDict.git
   ```

2. Install the required Python dependencies:

   ```bash
   pip install flask torch
   ```

3. Navigate to the app directoty:
   ```bash
   cd TrydoDict/app
   ```

4. Start the flask application:
   ```bash
   python main.py
   ```

   You can access the application via http://localhost:8080. Below is what the starting interface looked like.
   
   ![image](https://github.com/thassung/TrydoDict/assets/105700459/9745ad8c-a675-4f40-a675-bd11dae4c45c)


## Evaluation

### Loss and Perplexity

| Attention | Training Loss | Training PPL | Validation Loss | Validation PPL |
|----------|---------------|--------------|-----------------|----------------|
| Scaled Dot-product Attention | 4.297 | 73.505 | 5.336 | 207.663 |
| General Attention | 4.184 | 65.617 | 5.546 | 256.115 |
| Multiplicative Attention | 3.949 | 51.903 | 5.443 | 231.157 |
| Additive Attention | 5.361 | 212.896 | 3.821 | 45.644 | 

   The evaluation is from evaluations of ten epochs training on each model with different attention mechanisms. The loss and PPL (perplexity) of each model is picked from the epoch with the least validation loss and PPL which is also the evaluation of each model in [model](model). The epoch that yields the least validation loss for each model are

| Attention | Training Loss | Validation Loss | From Epoch | Total Epoch | Avg. Training Time per Epoch | Memory Usage |
|----------|---------------|--------------|-----------------|----------------|----------------|----------------|
| Scaled Dot-product Attention | 4.297 | 5.336 | 2 | 10 | 47.5 s | 4.89 GB |
| General Attention | 4.184 | 5.546 | 4 | 10 | 46.3 s | 4.89 GB |
| Multiplicative Attention | 3.949 | 5.443 | 5 | 10 | 47.2 s | 4.99 GB |
| Additive Attention | 5.361 | 3.821 | 5 | 6 | 2 s | 12.46 GB |

   The number of trained epoch for the MT with additive attention is only 6 because of the available memory is not sufficient to handle additional linear layers for more than 6 epochs. In addition, the number of data rows used to train the MT with additive attention is much smaller—200 rows for training, 40 rows for validation, and 40 rows for training. The comparison between the MT and the three other MTs should be done with this fact in mind that it will not be entirely appropriate to do so.

![loss vs epoch](./model/evaluation/losses.png)
![val_loss vs epoch](./model/evaluation/val_losses.png)

   Considering only the three models that are trained with the same size of dataset, the model with scaled dot-product attention mechanism has reached the smallest validation loss (the best performance) at its second epoch followed by multiplicative attention and general attention respectively. The computational loads are very close to each other (see avg. training time per epoch and memory usage in the above table). To some extent, because their losses and computational loads are similar, it can be said that the three models have similar computational efficiency even though the two models with no additional layer, scaled dot-product and general attention, are expected to use lower computional power. Here, additive attention aside is unarguably the worst in computational efficiency due to the astronomical memory usage.

### Attention

<table>
  <tr>
    <td>
      <img src="./model/evaluation/attention_scaled-dot-product.png" alt="attention vs epoch with scaled dot-product attention" width="400">
      <p>Scaled dot-product attention</p>
    </td>
    <td>
      <img src="./model/evaluation/attention_general.png" alt="attention vs epoch with general attention" width="400">
      <p>General attention</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="./model/evaluation/attention_additive.png" alt="attention vs epoch with additive attention" width="400">
      <p>Additive attention</p>
    </td>
    <td>
      <img src="./model/evaluation/attention_multiplicative.png" alt="attention vs epoch with multiplicative attention" width="400">
      <p>Multiplicative attention</p>
    </td>
  </tr>
</table>

The input text (English) and the target text (Thai) are

ENG: *FAR LEFT: Indonesian National Police Chief Tito Karnavian, from left, Philippine National Police Chief Ronald Dela Rosa and Royal Malaysian Police Inspector General Khalid Abu Bakar link arms before the Trilateral Security Meeting in Pasay city, southeast of Manila, Philippines, in June 2017. [THE ASSOCIATED PRESS]*

TH: *(ซ้ายสุด) นายติโต คาร์นาเวียน ผู้บัญชาการตํารวจแห่งชาติอินโดนีเซีย (จากซ้าย) นายโรนัลด์ เดลา โรซา ผู้บัญชาการตํารวจแห่งชาติฟิลิปปินส์ และนายคาลิด อาบู บาการ์ ผู้บัญชาการตํารวจแห่งชาติมาเลเซีย ไขว้แขนกันก่อนเริ่มการประชุมความมั่นคงไตรภาคีในเมืองปาเซย์ ซึ่งอยู่ทางตะวันออกเฉียงใต้ของกรุงมะนิลา ประเทศฟิลิปปินส์ ในเดือนมิถุนายน พ.ศ. 2560 ดิแอสโซซิเอทเต็ด เพรส*

   The attention heatmap can be interpreted by reading the query token (column-wise) and the key token (row-wise). The weight (color map gradient) probabilistically represents where attention is directed to when querying over a key. For example, for the MT with general attention, the word *Rosa* gives high attention to many output tokens especially *ภาพ* (means "picture" (noun.))implying these two words are closely correlated (for the model's perspective, at least).

   From the attention matrices, the multiplicative shows the most promising attention since it successfully learns to give attention to *แห่งชาติ* (means "national" (adj.)) when *National* is the query token. This is the reason why the model with multiplicative attention is used in the application. You can also change to the model trained with different attention by modifying *att_mech* parameter in *main.py* file in *app* folder at [this line of code](https://github.com/thassung/TrydoDict/blob/8e60c572dab0a95970c864f74102374bbd0f8397/app/main.py#L331-L335). However, there is only a few notable correct/expected attentions among the models. This might be the problem of the chosen subset of training dataset which involves a highly specific context and contains a lot of proper nouns which proves to be a challenging task for a small model to generalize and understand.
