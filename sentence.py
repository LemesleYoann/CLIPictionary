from model import encoder_text
import torch, clip, random
import numpy as np
device = torch.device("cpu")

from words import words

########## SENTENCE PART #######################################################
voyelles = ["a","e","i","o","u"]
links    = list(words.keys())[1:]

def link_text(part,nextWord):
    ### Check if we need to write "... a", "... an", "..."
    if (len(part["link"]) > 0) and (part["link"][-1] == "a"):
        voyelleStart = (nextWord[0] in voyelles)
        plural       = (nextWord[-1] == "s" and nextWord[-2] != "s") or (nextWord in ["nothing","hair","vampire teeth","something"])
    else:
        voyelleStart, plural = False, False
    return (part["link"][:-2] if plural else part["link"] + ("n" if voyelleStart else ""))

def part_text(part):
    l = link_text(part,part["word"])
    return l + (" " if len(l)>0 else "") + part["word"]

def compute_embeddings(part,var_dict,prefix,batch_size=64):
    target        = part["word"]
    possibleWords = list(set(words[part["link"]]) - set([target]+var_dict["found_words"]))
    if len(possibleWords) > (batch_size-1): possibleWords = np.random.choice(list(possibleWords),batch_size-1,replace=False).tolist()
    possibleWords.append(target)
    ### Compute all classes & embeddings for current sentence part
    part["classes"] = [prefix + link_text(part,w) + (" " if len(link_text(part,w))>0 else "") + w for w in possibleWords]
    with torch.no_grad():
        embeddings         = encoder_text(clip.tokenize(part["classes"]).to(device))
        embeddings        /= embeddings.norm(dim=-1, keepdim=True)
        part["embeddings"] = embeddings.tolist()

########## SENTENCE ############################################################
def iniSentence(var_dict,input="",first_game=False):
    var_dict["found_words"] = []
    var_dict["parts"]       = []
    var_dict["step"]        = 0
    prefix                  = ""
    N                       = 2

    if first_game:
        link = "a drawing of a"
        part = {"link":link,"word":"cat","classes":[],"embeddings":[]}
        var_dict["parts"].append(part)
        compute_embeddings(part, var_dict, prefix)
        prefix += part_text(part) + " "

        link = "with a"
        part = {"link":link,"word":"face","classes":[],"embeddings":[]}
        var_dict["parts"].append(part)
        compute_embeddings(part, var_dict, prefix)
        prefix += part_text(part) + " "
    else:
        ##### Generating Random Sentence
        link = "a drawing of a"
        part = {"link":link,"word":np.random.choice(words[link]),"classes":[],"embeddings":[]}
        var_dict["parts"].append(part)
        compute_embeddings(part, var_dict, prefix)
        prefix += part_text(part) + " "

        for i in range(N-1):
            link  = np.random.choice(links)
            part  = {"link":link,"word":np.random.choice(words[link][1:]),"classes":[],"embeddings":[]}
            var_dict["parts"].append(part)
            compute_embeddings(part, var_dict, prefix)
            prefix += part_text(part) + " "

    var_dict["target_sentence"] = prefix[:-1] # Target sentence is prefix without the last space
    setState(var_dict)
    return var_dict["target_sentence"]

def prevState(var_dict):
    if len(var_dict["prev_steps"]) > 0: var_dict["step"] = var_dict["prev_steps"].pop(-1)
    else:                               var_dict["step"] = 0
    var_dict["revertedState"] = True
    setState(var_dict)

def setState(var_dict):
    var_dict["found_words"] = var_dict["found_words"][:var_dict["step"]]
    var_dict["guessed_sentence"] = ""
    for i in range(var_dict["step"]):
        var_dict["guessed_sentence"] += part_text(var_dict["parts"][i]) + " "

def updateState(var_dict, preds):
    if not var_dict["revertedState"]:  var_dict["prev_steps"].append(var_dict["step"])
    else:                              var_dict["revertedState"] = False

    ### Check if the current part has been guessed
    part = var_dict["parts"][var_dict["step"]]

    '''
    idx_of_nothing = -1
    if ("nothing" in preds[0]):   idx_of_nothing = 0
    elif ("nothing" in preds[1]): idx_of_nothing = 1
    elif ("nothing" in preds[2]): idx_of_nothing = 2

    idx_of_guess = -1
    if (part["classes"][-1] == preds[0]):   idx_of_guess = 0
    elif (part["classes"][-1] == preds[1]): idx_of_guess = 1
    elif (part["classes"][-1] == preds[2]): idx_of_guess = 2
    '''

    if not var_dict["win"] and part["classes"][-1] in preds:
        var_dict["step"] += 1
        var_dict["found_words"].append(part["word"])
        var_dict["win"] = var_dict["step"] == len(var_dict["parts"])
        setState(var_dict)
        if var_dict["win"]: return 1
        else:               return 0
    elif not var_dict["win"]: return -1
    else:                     return 1
