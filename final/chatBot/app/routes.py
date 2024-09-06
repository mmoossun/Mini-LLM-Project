from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import requests
import re
from langchain_openai import AzureChatOpenAI

main = Blueprint('main', __name__)

exclude_place_ids = []
saved_routes = []  # 메모리에 저장된 경로

@main.route('/')
@login_required
def index():
    return render_template('index.html', api_key=current_app.config['GOOGLE_MAPS_API_KEY'])

def search_nearby_places(lat, lng, api_key):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 1000,
        "key": api_key,
        "language": "ko",
        "type": "point_of_interest"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    return []

def get_place_details(place_id, api_key):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,formatted_address,reviews,geometry",
        "key": api_key,
        "language": "ko"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('result', {})
    return {}

def summarize_places_with_gpt(prompt, config):
    model_name = config['AZURE_OPENAI_DEPLOYMENT']
    api_key = config['AZURE_OPENAI_API_KEY']
    api_version = config['OPENAI_API_VERSION']
    model = AzureChatOpenAI(
        azure_deployment=model_name,
        openai_api_key=api_key,
        api_version=api_version,
        temperature=0.7
    )
    model_output = model.invoke(prompt)
    content = model_output.content.strip()
    
    place_ids_1 = re.findall(r'- \*\*place_id\*\*:? *([A-Za-z0-9_-]+)', content)
    place_ids_2 = re.findall(r'- \*\*place_id:\*\*? *([A-Za-z0-9_-]+)', content)
    place_ids_3 = re.findall(r'place_id:\s*([A-Za-z0-9_-]+)', content)
    place_ids = place_ids_1 + place_ids_2 + place_ids_3
    
    return content, place_ids

@main.route('/search', methods=['POST'])
@login_required
def search():
    global exclude_place_ids
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    received_exclude_place_ids = data.get("exclude_place_ids", [])
    user_input = data.get("user_input", "")

    exclude_place_ids.extend(received_exclude_place_ids)

    try:
        nearby_places = search_nearby_places(lat, lng, current_app.config['GOOGLE_MAPS_API_KEY'])
        if not nearby_places:
            return jsonify({"error": "No nearby places found."}), 404

        nearby_places = [place for place in nearby_places if place['place_id'] not in exclude_place_ids]

        prompt = f"사용자의 입력: {user_input}\n\n사용자의 좌표는 ({lat}, {lng})입니다. 주어진 장소 정보는 {nearby_places}입니다. 주어진 장소 정보들 중에서 리뷰와 평점, 거리 등에 따라 그리고 사용자의 입력에 따른 3개의 장소를 추천해줘, 그리고 다음과 같은 정보를 간결하고 보기좋게 나열해줘 각각 항목 이후에 줄바꿈도 해줘 - 이름, 평점, 리뷰요약, 거리, 이유, 주소, place_id\n"
        
        summarized_info, place_ids,json_data = summarize_places_with_gpt(prompt, current_app.config)

        detailed_places = [get_place_details(place_id, current_app.config['GOOGLE_MAPS_API_KEY']) for place_id in place_ids]

        return jsonify({"places": detailed_places, "place_info": summarized_info, "model_output": summarized_info, "place_ids": place_ids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/save_route', methods=['POST'])
@login_required
def save_route():
    global saved_routes
    route_data = request.json
    saved_routes.append(route_data)
    return jsonify({"message": "Route saved successfully!"}), 200

@main.route('/get_routes', methods=['GET'])
@login_required
def get_routes():
    return jsonify(saved_routes), 200
