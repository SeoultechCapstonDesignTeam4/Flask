from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import math
try:
    vgg = models.vgg19_bn(weights=None)
    resNet = models.resnet101(weights=None)
    densNet = models.densenet121(weights=None)
    googlenet = models.googlenet(weights=None, aux_logits=False, init_weights=True)
    mobileNet = models.mobilenet_v2(weights=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_features = mobileNet.classifier[1].in_features
    mobileNet.classifier[1] = nn.Linear(num_features, 5)

    googlenet.fc = torch.nn.Linear(in_features=googlenet.fc.in_features, out_features=5, bias=True)

    densNet.classifier = nn.Linear(in_features=densNet.classifier.in_features, out_features=5, bias=True)

    resNet.fc = nn.Linear(in_features=resNet.fc.in_features, out_features=5, bias=True)

    vgg.classifier[6] = torch.nn.Linear(vgg.classifier[6].in_features, 5)
    densNet.load_state_dict(torch.load("denseNet.pt", map_location=torch.device('cpu')))
    resNet.load_state_dict(torch.load("resNet.pt", map_location=torch.device('cpu')))
    vgg.load_state_dict(torch.load("vgg.pt", map_location=torch.device('cpu')))
    mobileNet.load_state_dict(torch.load("mobileNet.pt", map_location=torch.device('cpu')))
    googlenet.load_state_dict(torch.load("googleNet.pt", map_location=torch.device('cpu')))
    # class EnsembleModel(torch.nn.Module):
    #     def __init__(self, models, class_names):
    #         super(EnsembleModel, self).__init__()
    #         self.models = models
    #         self.class_names = class_names

    #     def forward(self, x):
    #         # 각 모델의 예측을 저장할 리스트
    #         outputs = []
            
    #         # 각 모델의 예측을 구한다.
    #         for model in self.models:
    #             output = model(x)
    #             softmax_output = torch.nn.functional.softmax(output, dim=1)
    #             max_prob, predicted_index = torch.max(softmax_output, 1)
    #             predicted_class = self.class_names[predicted_index.item()]
                
    #             print(f"모델 {model.__class__.__name__}의 예측 결과: \n{predicted_class} (확률: {max_prob.item() * 100:.2f}%)")
    #             for class_index, class_prob in enumerate(softmax_output[0]):
    #                 print(f" - {self.class_names[class_index]} : {class_prob.item() * 100:.2f}%")
                
    #             outputs.append(softmax_output)
            
    #         # 예측의 평균을 구한다.
    #         avg_output = torch.mean(torch.stack(outputs), dim=0)
            
    #         return avg_output
    class EnsembleModel(torch.nn.Module):
        def __init__(self, models, class_names):
            super(EnsembleModel, self).__init__()
            self.models = models
            self.class_names = class_names

        def forward(self, x):
            # 각 모델의 예측을 저장할 리스트
            outputs = []
            predictions = []
            
            # 각 모델의 예측을 구한다.
            for model in self.models:
                output = model(x)
                softmax_output = torch.nn.functional.softmax(output, dim=1)
                max_prob, predicted_index = torch.max(softmax_output, 1)
                predicted_class = self.class_names[predicted_index.item()]
                
                model_predictions = {
                    "model": model.__class__.__name__,
                    "prediction": predicted_class,
                    "probability": round(max_prob.item() * 100,2),
                    "class_probabilities": {self.class_names[class_index]: round(class_prob.item() * 100,2) for class_index, class_prob in enumerate(softmax_output[0])}

                }
                predictions.append(model_predictions)
                
                outputs.append(softmax_output)
            
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            _, ensemble_prediction = torch.max(avg_output, 1)
            
            return predictions, self.class_names[ensemble_prediction.item()]
    class_names = {0: '결막염', 1: '백내장', 2: '색소침착성 각막염', 3: '유루증', 4: '정상'}

    # ensemble_model = EnsembleModel([googlenet, mobileNet, resNet, densNet], class_names)
    ensemble_model = EnsembleModel([googlenet, mobileNet, vgg, resNet, densNet], class_names)
except Exception as e:
    print(e)


app = Flask(__name__)

def preprocess_image(image_path):
    # 이미지를 열고 RGB 형식으로 변환
    image = Image.open(image_path).convert("RGB")
    
    # 이미지 크기를 조정 (224x224로 설정)
    image = transforms.Resize((224, 224))(image)
    
    # 이미지를 텐서로 변환
    image_tensor = transforms.ToTensor()(image)
    
    # 이미지 텐서를 모델에 맞게 정규화 (예: ImageNet의 평균 및 표준 편차)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)
    
    # 배치 차원을 추가하여 모델의 예상 입력 형식으로 조정
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


image_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

@app.route('/test', methods=['GET'])
def test():
    return "test"

import traceback


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        # 이미지 전처리 함수를 사용하여 이미지를 텐서로 변환
        image_tensor = preprocess_image(image_file)
        predictions, ensemble_prediction = ensemble_model(image_tensor)
        
        response = {
            "ensemble_prediction": ensemble_prediction,
            "model_predictions": predictions
        }

        return jsonify(response)
        # image_file = request.files['image']
        # # 이미지 전처리 함수를 사용하여 이미지를 텐서로 변환
        # image_tensor = preprocess_image(image_file)
        # output = ensemble_model(image_tensor)
        # # 예측 결과에서 가장 높은 확률을 가진 클래스를 선택한다.
        # _, predicted = torch.max(output, 1)
        
        # # 결과 출력
        # predicted_class = class_names[predicted.item()]
        # return jsonify(predicted_class)

    except Exception as e:
        tb = traceback.format_exc()
        print(e)
        return jsonify({'error': str(e), 'traceback': tb})
    

if __name__ == '__main__':
   app.run('0.0.0.0', port=5001, debug=True)