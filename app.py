import matplotlib.pyplot as plt
import torch, torchvision, clip, random
import random, math, time, os
from PIL import Image
import gradio as gr
import numpy as np

from sentence import *
from game import *

##### Initialize new game
title, _,_,var_dict    = new_game(first_game=True)
var_dict["start_time"] = -1

##### Display & Events
demo = gr.Blocks()
with demo:
    ### All game variables are stored here
    variables = gr.Variable(var_dict)
    ### Target Sentence
    title = gr.HTML(title)
    ### Canvas & Prediction
    with gr.Column():
        with gr.Row():
            image_input = gr.Image(image_mode='L', label="", show_label=False, source='canvas', shape=None, streaming=False, invert_colors=False, tool="editor")
            with gr.Column():
                html_pred    = gr.HTML(value=getHTML(var_dict,""))
                html_loading = gr.HTML("")
        ### 'New Sentence' Button
        button  = gr.Button("New Sentence",variant="primary")
        ### Informations
        gr.HTML("<div style=\"display:block; height:30px;\"> </div>")
        with gr.Row():
            gr.HTML("<div style=\"display:block; position:relative; bottom:10%; border-top: 1px solid grey;  padding:10px; \"><span style=\"font-size:30px;\">✏️</span><span style=\"font-size:40px; font-weight:bold;\">CLIPictionary!</span><br>Draw to make the model guess the target sentence displayed at the top!<br>Made by <a href=\"https://yoann-lemesle.notion.site/Yoann-Lemesle-63b8120764284794b275d2967be710da\" style=\"text-decoration: underline;\">Yoann Lemesle</a> using OpenAI's <a href=\"https://github.com/openai/CLIP\" style=\"text-decoration: underline;\">CLIP model</a>.</div>")


    ### Events
    button.click(loading,inputs=html_loading,outputs=[title,html_pred,html_loading])  # Button -> triggers Loading
    html_loading.change(new_game,inputs=[html_loading],outputs=[title,html_pred,image_input,variables])     # Loading -> triggers New game
    image_input.change(process_img, inputs=[variables,image_input,title], outputs=[html_pred,title,variables])


demo.launch(share=False)
