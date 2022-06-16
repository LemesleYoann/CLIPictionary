################################################################################
# A DRAWING OF A ...
shapes      = ["triangle","square","circle","heart","star","diamond"]
animals     = ["cat","dog","duck","bee","butterfly","bird","pig","cow","fish","frog","shark","snake","mouse","monkey","snail"]
objects     = ["wine glass","eye","plane","spoon","basket ball","chair","pen","computer","hat","soccer ball","phone","sword","axe","umbrella","bell","dumbbell","scissors","fork","bag","clock","key","shopping cart","car","boat","house","mug","sun","moon","atom","hand"]
plants      = ["tree","flower","leaf","palm tree","mushroom"] # I know that mushrooms are not plants stop coming to my house
food        = ["donut","coconut","banana","apple","bottle","sausage","icecream","burger","egg","lollypop","pizza"]
instruments = ["drum","guitar","piano","flute","trumpet","accordion"]

# FEELING ...
feelings = ["neutral","happy","sad","angry","surprised","thirsty","sleepy","hungry","love","curious","evil"]

# WITH A...
attributes = ["face","mustache","muscles","vampire teeth","hair","eye","mouth","scar"]

# WEARING A...
clothings  = ["glasses","sunglasses","hat","socks","eye patch","pants","tee-shirt","scarf"]

# ...
others = ["that is talking","that is dancing","that is singing","and its clone","at the beach","at a forest"]

words = {
    "a drawing of a": shapes + animals + objects + instruments + plants + food,
    "with a":         ["nothing"]+attributes + clothings,
    "wearing a":      ["nothing"]+clothings,
    "eating a":       ["nothing"]+food,
    "playing the":    ["nothing"]+instruments,
    "holding a":      ["nothing"]+shapes + animals + objects + instruments + plants + food,
    "feeling":        ["nothing"]+feelings,
    "":               ["and nothing else"]+others
    }
################################################################################
