#This code is giving me only Batch size output


#Loading Model
model = my_model.get_model()
model.load_state_dict(torch.load(os.path.join(config.output_path(), "baseline.h5")))

#loading test data
test_data = preprocess.load_test_data()
print(len(test_data)) //14000

for i, data in enumerate(test_data):
    data = data.unsqueeze(1)
    output = model(data)
    preds = output.cpu().data.max(1, keepdim=True)[1] 
print(len(output)) //2



#This code stucked 2H (why?)

#Loading Model
model = my_model.get_model()
model.load_state_dict(torch.load(os.path.join(config.output_path(), "baseline.h5")))

#loading test data
test_data = preprocess.load_test_data()
print(len(test_data)) //14000

final_output = []
for i, data in enumerate(test_data):
    data = data.unsqueeze(1)
    output = model(data)
    final_output.append(output)
result = torch.cat(final_output, dim=1) 

print(len(result))