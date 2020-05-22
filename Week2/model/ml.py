from monkeylearn import MonkeyLearn

ml = MonkeyLearn('1408f39bdb6710d995721763d66a7232da810e2d')
model_id = 'cl_aLJEoL8A'
tags_id = ml.classifiers.detail(model_id).body["model"]["tags"]

for tag_id in tags_id:
    print("Result " + tag_id["name"] + ":")
    response = ml.classifiers.tags.detail(model_id, tag_id["id"]).body["stats"]
    del response["keywords"]
    print(response, "\n")

data = [input("Input: ")]
result = ml.classifiers.classify(model_id, data).body[0]['classifications'][0]['tag_name'].upper()
print("Output: ", result)