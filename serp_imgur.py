import requests
import os
from dotenv import load_dotenv


load_dotenv()
# Imgur and SERP API credentials
imgur_client_id = os.getenv('imgur_client_id')
serp_api_key = os.getenv('serp_api_key')
search_endpoint = 'https://serpapi.com/search'


def upload_image_to_imgur(image_path):
    headers = {'Authorization': f'Client-ID {imgur_client_id}'}
    data = {'image': open(image_path, 'rb').read()}
    response = requests.post('https://api.imgur.com/3/image', headers=headers, files=data)
    response_data = response.json()
    if response.status_code == 200 and response_data['success']:
        return response_data['data']['link']
    else:
        raise Exception(f"Error uploading image to Imgur: {response_data['data']['error']}")

def reverse_image_search(image_url):
    params = {
        'engine': 'google_reverse_image',
        'image_url': image_url,
        # "image_content": image_url,
        'api_key': serp_api_key
    }
    response = requests.get(search_endpoint, params=params)
    return response.json()

def extract_titles_and_descriptions(search_results, top_n=3):
    titles_and_descriptions = []
    for result in search_results.get('image_results', [])[:top_n]:
        temp_dict = {}
        title = result.get('title', '')
        description = result.get('snippet', '')
        temp_dict['title'] = title
        temp_dict['description'] = description
        titles_and_descriptions.append(temp_dict)
    return titles_and_descriptions

def main(image_path):
    # try:
        # Upload image to Imgur and get the URL
        image_url = upload_image_to_imgur(image_path)
        print(f"Image uploaded to Imgur: {image_url}")

        # Perform reverse image search
        search_results = reverse_image_search(image_url)
        if 'error' in search_results:
            print("Error:", search_results['error'])
            return

        # Extract titles and descriptions
        titles_and_descriptions = extract_titles_and_descriptions(search_results)
        print(titles_and_descriptions)
        # Print results
        # for idx, (title, description) in enumerate(titles_and_descriptions):
        #     print(f"Result {idx+1}:")
        #     print("Title:", title)
        #     print("Description:", description)
        #     print("-" * 50)
    # except Exception as e:
    #     print(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = 'sampleImages/edited3.jpg'  # Replace with the path to your local image
    main(image_path)
