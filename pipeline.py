import requests
import base64
from langchain_core.output_parsers import JsonOutputParser
import base64
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
# Imgur and SERP API credentials
imgur_client_id = os.getenv('imgur_client_id')
serp_api_key = os.getenv('serp_api_key')
search_endpoint = 'https://serpapi.com/search'

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('gpt_api_key')
# Replace with your OpenAI API key
gpt_api_key = os.getenv('gpt_api_key')


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

def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}

class ImageInformation(BaseModel):
    """Information about an image."""

    Title: str = Field(description="Suitable title for the given product in image")
    image_description: str = Field(description="a short description of the image")
    #  main_objects: list[str] = Field(description="list of the main objects on the picture")


# Set verbose
# globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4-vision-preview", max_tokens=1024)
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
                ])]
                )
    return msg.content


load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)

parser = JsonOutputParser(pydantic_object=ImageInformation)
def get_image_informations(image_path: str) -> dict:
    vision_prompt = """
    Given the image, the image is a commercial product. I want to get the information for listing this product on online store. provide the following information:
    - The extracted text written on the product.
    - Title of the product in image based on the extracted text
    """
    vision_chain = load_image_chain | image_model | parser
    return vision_chain.invoke({'image_path': f'{image_path}', 
                                'prompt': vision_prompt})

def parse_json_response(response):
    # Remove the enclosing markers if present
    if response.startswith("```json") and response.endswith("```"):
        response = response[7:-3].strip()
    
    # Load the response as a JSON object
    data = json.loads(response)
    
    # Find the key that contains the list of items
    listings_key = None
    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            listings_key = key
            break
    
    if not listings_key:
        raise ValueError("No valid listings key found in the response")
    
    listings = data[listings_key]
    
    # Create a list to store the parsed dictionaries
    parsed_data = []
    
    # Iterate through each item in the listings
    for item in listings:
        # Extract the title and features
        title = item.get("Title", "")
        features = item.get("Features", [])
        
        # Create a dictionary for each item
        item_dict = {
            "Title": title,
            "Features": features
        }
        
        # Append the dictionary to the list
        parsed_data.append(item_dict)
    
    return parsed_data

def main(image_path):
    # try:
        # Upload image to Imgur and get the URL
        image_url = upload_image_to_imgur(image_path)
        print(f"Image uploaded to Imgur: {image_url}")

        # Perform reverse image search
        search_results = reverse_image_search(image_url)
        if 'error' in search_results:
            print("Error in Serp API:", search_results['error'])
        

        # Extract titles and descriptions
        serp_results = extract_titles_and_descriptions(search_results)
        print("Serp Result: ",serp_results, "\n\n\n\n")

        gpt_vision_result = get_image_informations(image_path)
        print("GPT Vision Result: ", gpt_vision_result, "\n\n\n\n")


        # Prompt to generate the JSON for the product listing
        prompt = f'''
        You have results from a SERP API and GPT Vision. The SERP API provides related product information, while GPT Vision gives exact extracted texts and a suitable title for the product image.
        Your task is to generate titles and feature lists for an e-commerce listing in JSON format. Prioritize the accurate GPT Vision data, using SERP API data ONLY if it is relevent to GPT Vision result. 
        #### SERP Results:
        {serp_results}

        #### GPT Vision Result:
        {gpt_vision_result}

        #### Example JSON format for one product:
        ```json
        {{
            "Listings": [
                {{
                    "Title": "Example Title",
                    "Features": [
                        "Feature 1",
                        "Feature 2",
                        "Feature 3",
                        .,
                        .,
                        .,
                        .,
                        .,
                        "feature N"
                    ]
                }}
            ]
        }}
        Generate a JSON for product listing (at Least THREE) based on the above results:
        '''

        gpt_model = OpenAI(api_key=gpt_api_key)
        # Call the ChatGPT 3.5 model using the chat completion endpoint
        response = gpt_model.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        # Extract the text from the response
        generated_text = response.choices[0].message.content

        print("Generated Text: ",generated_text)
        parsed_data = parse_json_response(generated_text)
        # Print the ChatGPT response

        return parsed_data

if __name__ == "__main__":
    image_path = 'sampleImages/edited3.jpg'  # Replace with the path to your local image
    main(image_path)
