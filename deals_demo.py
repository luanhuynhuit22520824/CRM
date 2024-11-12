from flask import Flask, request, jsonify
import requests
from transformers import pipeline, MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Cấu hình mô hình dịch và phân tích cảm xúc
model_name = 'Helsinki-NLP/opus-mt-vi-en'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Hàm dịch tiếng Việt sang tiếng Anh
def translate_to_english(text):
    translated = tokenizer(text, return_tensors="pt", padding=True)
    translated_text = model.generate(**translated)
    return tokenizer.decode(translated_text[0], skip_special_tokens=True)

# Thông tin xác thực API của HubSpot
api_token = 'YOUR_HUBSPOT_API_TOKEN'
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json

    # Kiểm tra dữ liệu form submission từ HubSpot
    values = data.get("values", [])
    firstname = ""
    how_can_i_help_you_ = ""
    email = ""

    # Lấy dữ liệu form
    for item in values:
        if item.get("name") == "firstname":
            firstname = item.get("value", "").strip()
        elif item.get("name") == "how_can_i_help_you_":
            how_can_i_help_you_ = item.get("value", "").strip()
        elif item.get("name") == "email":
            email = item.get("value", "").strip()

    # Dịch yêu cầu từ tiếng Việt sang tiếng Anh
    translated_text = translate_to_english(how_can_i_help_you_) if how_can_i_help_you_ else ""
    # Phân tích cảm xúc của yêu cầu
    sentiment = classifier(translated_text)[0] if translated_text else None

    # Tạo tên deal từ thông tin khách hàng
    dealname = f"{firstname} - {email} - {how_can_i_help_you_}"
    deal_data = {
        "properties": {
            "dealname": dealname,
        }
    }

    # Thiết lập pipeline và deal stage dựa trên phân tích cảm xúc
    if not how_can_i_help_you_:
        deal_data["properties"]["pipeline"] = "635966392"  # ID của Lead Pipeline
        deal_data["properties"]["dealstage"] = "938431356"  # ID của Stage "Chưa giao người xử lý"
    elif sentiment and sentiment['label'] == 'POSITIVE':
        deal_data["properties"]["pipeline"] = "default"  # ID của Sale Pipeline
        deal_data["properties"]["dealstage"] = "appointmentscheduled"  # ID của Stage "Mới"
    elif sentiment and sentiment['label'] == 'NEGATIVE':
        deal_data["properties"]["pipeline"] = "635966392"  # ID của Lead Pipeline
        deal_data["properties"]["dealstage"] = "938431362"  # ID của Stage "Cơ hội rác"

    # Tạo deal trong HubSpot
    deal_url = "https://api.hubapi.com/crm/v3/objects/deals"
    response = requests.post(deal_url, headers=headers, json=deal_data)

    if response.status_code == 201:
        return jsonify({"status": "success", "message": "Deal created successfully!"}), 201
    else:
        return jsonify({"status": "error", "message": "Failed to create deal"}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
