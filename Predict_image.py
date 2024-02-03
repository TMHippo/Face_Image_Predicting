import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn 
import io
import flask_app as app

class CelebModel(nn.Module):
    def __init__(self,num_classes=40):
        super(CelebModel,self).__init__()

        self.unit1=Layer(in_ch=3,out_ch=32)
        self.unit2=Layer(in_ch=32,out_ch=32)
        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.unit3=Layer(in_ch=32,out_ch=64)
        self.unit4=Layer(in_ch=64,out_ch=64)
        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.unit5=Layer(in_ch=64,out_ch=128)
        self.unit6=Layer(in_ch=128,out_ch=128)
        self.unit7=Layer(in_ch=128,out_ch=128)
        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.unit8=Layer(in_ch=128,out_ch=256,kernel_size=5,padding=0)
        self.unit9=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.unit10=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.pool4=nn.MaxPool2d(kernel_size=2)

        self.drop2=nn.Dropout(0.5)

        self.unit11=Layer(in_ch=256,out_ch=512,kernel_size=3,padding=0)
        self.unit12=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)
        self.unit13=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)

        self.pool5=nn.AvgPool2d(kernel_size=2)

        self.drop3=nn.Dropout(0.5)

        self.model=nn.Sequential(self.unit1,self.unit2,self.pool1,self.unit3,
                                 self.unit4,self.pool2,self.unit5,self.unit6,
                                 self.unit7,self.pool3,self.unit8,self.unit9,
                                 self.unit10,self.pool4,self.drop2,self.unit11,
                                 self.unit12,self.unit13,self.pool5,self.drop3)

        self.fc=nn.Linear(in_features=512,out_features=num_classes)


    def forward(self,Input):

        output=self.model(Input)
        output=output.view(-1,512)
        output=self.fc(output)

        return output
    
# Defining Class for Single Layer.
class Layer(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
      """ in_ch: Số lượng kênh đầu vào input channels.
      out_ch: Số lượng kênh đầu ra output channels.
      kernel_size: Kích thước của hạt nhân tích chập convolutional kernel (mặc định 3x3).
      stride: Bước nhảy của phép toán tích chập convolution operation (mặc định 1).
      padding: Phần đệm được thêm vào (mặc định 1). """
      super(Layer,self).__init__()
      self.conv=nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding)
      self.bn=nn.BatchNorm2d(out_ch) # Thêm một lớp chuẩn hóa theo loạt với số lượng tính năng bằng `out_ch`.
      self.relu=nn.ReLU() # Thêm một hàm kích hoạt đơn vị tuyến tính chỉnh lưu (ReLU).
      nn.init.xavier_uniform_(self.conv.weight) # Khởi tạo các trọng số của lớp tích chập bằng cách sử dụng khởi tạo Xavier/Glorot.

    def forward(self,Input):
        output=self.conv(Input) # Áp dụng phép toán tích chập cho đầu vào
        output=self.bn(output) # Áp dụng chuẩn hóa theo loạt cho đầu ra của phép tích chập.
        output=self.relu(output) # Áp dụng hàm kích hoạt ReLU cho đầu ra đã được chuẩn hóa theo loạt.
        return output
    
# Load the model
model = torch.load('model.pth', map_location=torch.device('cpu'))
model = model.eval()

# Transformation function
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image)

# Prediction function
def Predict(image):
    pred=model(image.unsqueeze(0))
    labels=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    attr=list(torch.round(torch.sigmoid(pred)).cpu().detach().numpy().squeeze(0))
    prd=list(torch.sigmoid(pred).cpu().detach().numpy().squeeze(0))
    new_labels=[label for label,a in list(zip(labels,attr)) if a==1]
    pred_list=[p for p,a in list(zip(prd,attr)) if a==1]
    return [{'name':label ,'percent':round(p,2)} for label,p in zip(new_labels,pred_list)]

def main():
    st.title('Celeb Attribute Predictor')

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Transform and predict when an image is uploaded
        image_bytes = uploaded_file.read()
        image = transform_image(image_bytes)
        predictions = Predict(image)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display predictions in the first column
        with col1:
            st.subheader('Predictions:')
            for prediction in predictions:
                st.write(f"{prediction['name']}: {prediction['percent']}")

        # Display the uploaded image in the second column
        with col2:
            st.subheader('Uploaded Image:')
            st.image(Image.open(io.BytesIO(image_bytes)), caption='Uploaded Image.', use_column_width=True)

if __name__ == "__main__":
    main()