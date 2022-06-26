import torch, torchvision, clip, time, math
import matplotlib.pyplot as plt
from model import encoder_image
from sentence import *

##### Get infos & cool facts to display during loadings
with open("infos.txt") as file:
    infos = file.readlines()

##### Get css
with open("style.css") as style:
    css = "<style>"+ ''.join(style.readlines())+"</style>"

##### 'DIFFICULTY SWITCH' EVENT
def switch_difficulty(var_dict, html_loading):

    var_dict["difficulty"] = 1 - var_dict["difficulty"]

    title, infos, new_value = loading(html_loading)
    return var_dict, title, infos, new_value

##### 'LOADING' EVENT
def loading(html_loading=None):
    ### This is just to make sure the content changes, which triggers the .change event which, itself, will launch a new game
    if html_loading == "<div style=\"display:none;\">0</div>": new_value = "<div style=\"display:none;\">1</div>"
    else:                                                      new_value = "<div style=\"display:none;\">0</div>"
    ### Get a random tip
    info = np.random.choice(infos)
    ### Return TITLE, TIP TEXT, NEW HTML CONTENT, CANVAS IMG
    return "<h1 id=\"loading\">⌛Loading...</h1>",css+"<div id=\"prediction\"><p id=\"infos\">"+info+"</p></div>",new_value

##### 'NEW GAME' EVENT
def new_game(var_dict,img=None,first_game=False):
    print("\n----------Launching new game!")

    if None is not var_dict:    difficulty = var_dict["difficulty"]
    else:                       difficulty = 1

    var_dict = {
        "start_time":           time.time(),
        "total_time":           0,
        "found_words":          [],
        "target_sentence":      "",
        "guessed_sentence":     "",
        "parts":                [],
        "win":                  0,
        "step":                 0,
        "prev_steps":           [],
        "prev_norm":            float("inf"),
        "tip":                  "",
        "loading":              False,
        "revertedState":        False,
        "difficulty":           difficulty
    }
    target = iniSentence(var_dict,first_game=first_game)
    ### Return TITLE, PREDICTION TEXT, CANVAS IMG, VAR DICT
    return "<h1>"+target+"</h1>", getHTML(var_dict,""), None, var_dict

##### PREDICTION TEXT HTML
def getHTML(var_dict,text,win=0):
    ### Which parts of the sentence have been guessed?
    guessed, not_guessed = "", ""
    text_words           = text.split(" ")
    target_words         = var_dict["target_sentence"].split(" ")
    for i,word in enumerate(text_words):
        if i < len(target_words) and word == target_words[i]: guessed += word + " "
        else:                                                 not_guessed += word + " "
    ### Display prediction
    if win!=1:
        html = "<p><span>"+guessed+"</span>"+not_guessed+"</p>"
    else:
        minutes, seconds  = math.floor(var_dict["total_time"]/60), var_dict["total_time"]%60
        if minutes < 1 and seconds <= 30:   emoji = "🏆😍"
        elif minutes < 1:                   emoji = "😄"
        elif minutes < 2:                   emoji = "😐"
        elif minutes < 3:                   emoji = "😓"
        else:                               emoji = "😱"
        time_str = "Total time: "+ ((str(minutes)+"m") if minutes>0 else "") + str(seconds)+"s "+emoji
        html     = "<p id=\"win\"><span>"+guessed+"</span><br>"+time_str+"</p>"
    return css+"<div id=\"prediction\">"+html+"</div>"

##### DRAWING PROCESSING & GAME STATE UPDATE
def process_img(var_dict,img,title):
    # Makes sure that start_time is updates for the first game
    if var_dict["start_time"] == -1:
        var_dict["start_time"] = time.time()
    if (None is img):
        return getHTML(var_dict,"",win=0),"<h1>"+var_dict["target_sentence"]+"</h1>",var_dict
    elif (None is not img) and (var_dict["win"] != 1):
        print("-----Processing...")
        part   = var_dict["parts"][var_dict["step"]]
        image = torch.tensor(img).float() / 255

        ### Detect Cancel event
        norm  = torch.norm(image)
        if norm > var_dict["prev_norm"]:
            print("---Cancel Event")
            prevState(var_dict)
        var_dict["prev_norm"] = norm

        ### Image preprocessing --> shape (224,224)
        max_edge = max(image.shape[0],image.shape[1])
        min_edge = min(image.shape[0],image.shape[1])
        square_image  = torch.ones(max_edge,max_edge)
        pad           = math.floor((max_edge - min_edge)/2)
        if max_edge == image.shape[1]: square_image[pad:pad+min_edge,:] = image
        else:                          square_image[:,pad:pad+min_edge] = image
        image = torchvision.transforms.Resize((224,224))(square_image.unsqueeze(0)).repeat(1,3,1,1)

        ### Computing cosine similarities (drawing<->text embeddings)
        with torch.no_grad():
            image_features = encoder_image(image)[0]
            text_features  = torch.tensor(part["embeddings"])
            image_features /= image_features.norm()
            similarities   = torch.matmul(text_features,image_features)
            probs          = torch.nn.Softmax(dim=-1)(similarities)

        ### Sort indexes by similarity
        idxs   = np.argsort(similarities)

        ### Use top-3 preditions
        top3_idxs = idxs[-3:]
        classes   = part["classes"]
        preds     = [classes[idx] for idx in top3_idxs]
        print(f"Top-3 Predictions: {preds}")
        print(f"Top-3 Probabilities: {probs[top3_idxs]}")

        ### Check if win (-1: bad guess, 0:progress=guessed sentence part, 1:win=guessed whole sentence)
        win = updateState(var_dict, preds)
        if win == -1:
            text = preds[-1]
        elif win == 0:
            part = var_dict["parts"][var_dict["step"]]
            text = var_dict["guessed_sentence"] + link_text(part,"something") + " something"
        elif win == 1:
            text = var_dict["guessed_sentence"]
            if var_dict["total_time"] == 0: var_dict["total_time"] = round(time.time() - var_dict["start_time"])
        return getHTML(var_dict,text,var_dict["win"]),"<h1>"+var_dict["target_sentence"]+"</h1>",var_dict
    else:
        return getHTML(var_dict,var_dict["target_sentence"],win=1),"<h1>"+var_dict["target_sentence"]+"</h1>",var_dict
