import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import requests
from io import BytesIO

class ImageStoryGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device) 
        
        self.story_generator = pipeline("text-generation", model="roneneldan/TinyStories-33M", device=-1)
        
        print("Models loaded and ready!")

    def load_image(self, image_path):

        if image_path.startswith('http'):  
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:  
            image = Image.open(image_path)
        return image.convert('RGB')  

    def get_caption(self, image):
   
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)  
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=50, num_beams=3)
        return self.caption_processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    def create_story(self, caption, length="medium"):
    
        prompt = f"This is a story about {caption}. One day,"
        max_tokens = {"short": 100, "medium": 150, "long": 200}[length]
        
        result = self.story_generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            return_full_text=False
        
        )
        story = result[0]['generated_text'].strip()

        while not story.endswith((".", "!", "?")):
            additional_result = self.story_generator(
                story,
                max_new_tokens=20, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                return_full_text=False
            )
            additional_text = additional_result[0]['generated_text'].strip()
            story += " " + additional_text
        
        return story
    def generate_from_image(self, image_path, length="medium"):
       
        try:
            image = self.load_image(image_path)
            caption = self.get_caption(image)
            story = self.create_story(caption, length)
            return {
                "caption": caption,
                "story": story,
                "word_count": len(story.split()),
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    print("Image Story Generator")
    print("Using TinyStories model for narrative generation")

    generator = ImageStoryGenerator()
    
    while True:
        image_path = input("\nImage path/URL (or 'quit'): ").strip()
        if image_path.lower() in ['quit', 'q', 'exit']:
            break
        
        length = input("Length (short/medium/long) [medium]: ").strip().lower()
        if length not in ['short', 'medium', 'long']:
            length = 'medium'
        
        print("Generating story")
        result = generator.generate_from_image(image_path, length)
        
        if result["success"]:
            print("\n" + "=" * 50)
            print(f"Caption: {result['caption']}")
            print(f"Words: {result['word_count']}")
            print(f"\nStory:\n{result['story']}")
            print("=" * 50)
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()