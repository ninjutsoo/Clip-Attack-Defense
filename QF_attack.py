import numpy as np
import torch
import random
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
device = 'cuda'
from attack import search_min_sentence_iteration, genetic, PGDattack, get_char_table, train
# from defense import search_min_sentence_iteration, genetic, PGDattack, get_char_table, train
len_prompt = 5

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to(device)
char_table = get_char_table()


attack_sentence = "a snake and a young man"

# def getPromptByGreedy():
#     #  Greedy
#     greedy_sentence = search_min_sentence_iteration(attack_sentence, char_table, len_prompt,
#                                                 1, tokenizer=tokenizer, text_encoder=text_encoder)
#     return greedy_sentence


# def getPromptByGenetic():
#     #  Genetic
#     prompts = []
#     for i in range(5):
#         genetic_prompt = genetic(attack_sentence, char_table, len_prompt, tokenizer=tokenizer,
#                                     text_encoder=text_encoder)
#         genetic_sentence = attack_sentence + ' ' + genetic_prompt[0][0]
#         prompts.add(genetic_sentence)
#     return prompts


#Defense
from defense import search_min_sentence_iteration, genetic, PGDattack, get_char_table, train

def getPromptByPGD():
    #  PGD
    prompts = []
    max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence,
                                                        len_prompt=len_prompt, char_list=char_table,
                                                        model=text_encoder.text_model, iter_num=100,
                                                        eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder)
    pgd_sentence = attack_sentence + ' ' + pgd_prompt
    print(pgd_sentence)
    # assuming loss_list is your list of losses
    loss_list = [x.item() for x in loss_list]
    loss_list = np.asarray(loss_list)
    loss_list = np.ones_like(loss_list) / loss_list
    
    # Create a simple list
    data = loss_list

    # Create a simple line plot
    plt.plot(data, 'o-', color='blue')

    # Find the maximum point and its index
    ymax = np.max(data)
    xmax = np.argmax(data)

    # Highlight the maximum point with a red dot
    plt.plot(xmax, ymax, 'ro')

    # Annotate the maximum point
    plt.annotate('max: ({}, {})'.format(xmax, ymax), (xmax, ymax), textcoords="offset points", xytext=(0,10), ha='center', color='red')

    # Add labels and title
    plt.xlabel('X-Axis Label')
    plt.ylabel('Y-Axis Label')
    plt.title('Plot Title')

    # Display the plot
    plt.show()
    return max_loss  

max_loss = getPromptByPGD()



def getPromptByPGD():
    #  PGD
    prompts = []
    max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence,
                                                        len_prompt=len_prompt, char_list=char_table,
                                                        model=text_encoder.text_model, iter_num=1,
                                                        eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder)
    pgd_sentence = attack_sentence + ' ' + pgd_prompt

    return max_loss


if __name__ == '__main__':
    max_loss = getPromptByPGD()
    print(max_loss)
