import kivy
from fontTools.ttLib.tables.C_P_A_L_ import Color
from gradio.themes.builder_app import sizes
from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import LabelBase
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from win32gui import Rectangle

# 加载模型和tokenizer
model = BertForSequenceClassification.from_pretrained('./my_model')
tokenizer = BertTokenizer.from_pretrained('./my_model')
LabelBase.register(name='SimHei', fn_regular='./simhei.ttf')
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 文本分类函数
def predict(text):
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "诈骗信息" if prediction == 0 else "正常短信"


# Kivy GUI
class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical',padding=10,spacing=10)
        with layout.canvas.before:
            Color(0.2,0.3,0.6,1)

        # 创建输入框
        self.text_input = TextInput(font_name = 'Simfang',hint_text='输入文本', size_hint=(1, 0.2), multiline=False)
        layout.add_widget(self.text_input)

        # 创建按钮
        self.predict_button = Button(text='预测', font_name = 'Simfang',size_hint=(1, 0.2))
        self.predict_button.bind(on_press=self.on_predict)
        layout.add_widget(self.predict_button)

        # 创建标签显示结果
        self.result_label = Label(text='预测结果将显示在这里',font_name = 'Simfang', size_hint=(1, 0.6))
        layout.add_widget(self.result_label)

        return layout



    def on_predict(self, instance):
        text = self.text_input.text  # 获取输入文本
        result = predict(text)  # 调用预测函数
        self.result_label.text = f'预测结果: {result}'  # 更新显示结果


if __name__ == '__main__':
    MyApp().run()
