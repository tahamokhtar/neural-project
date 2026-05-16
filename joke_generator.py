import requests
import json

def get_random_joke():
    """
    Fetches a random joke from the JokeAPI external API.
    
    Returns:
        dict: A dictionary containing the joke details
    """
    try:
        # Using JokeAPI - free and no authentication required
        url = "https://v2.jokeapi.dev/joke/Any"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise exception for bad status codes
        
        joke_data = response.json()
        
        if joke_data['type'] == 'single':
            # Single-line joke
            joke = {
                'type': 'single',
                'joke': joke_data['joke'],
                'category': joke_data['category']
            }
        else:
            # Two-part joke (setup + delivery)
            joke = {
                'type': 'twopart',
                'setup': joke_data['setup'],
                'delivery': joke_data['delivery'],
                'category': joke_data['category']
            }
        
        return joke
    
    except requests.exceptions.RequestException as e:
        return {'error': f"Failed to fetch joke: {str(e)}"}
    except json.JSONDecodeError:
        return {'error': "Failed to parse response"}


def display_joke(joke):
    """
    Displays the joke in a formatted way.
    
    Args:
        joke (dict): The joke data to display
    """
    if 'error' in joke:
        print(f"❌ {joke['error']}")
        return
    
    print("\n" + "="*50)
    print(f"📚 Category: {joke['category']}")
    print("="*50)
    
    if joke['type'] == 'single':
        print(f"😂 {joke['joke']}")
    else:
        print(f"🎭 Setup: {joke['setup']}")
        print(f"😄 Delivery: {joke['delivery']}")
    
    print("="*50 + "\n")


def main():
    """
    Main function to run the joke generator.
    """
    print("🎪 Welcome to the Random Joke Generator!")
    print("Powered by JokeAPI\n")
    
    while True:
        print("Options:")
        print("1. Get a random joke")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            joke = get_random_joke()
            display_joke(joke)
        elif choice == '2':
            print("\n👋 Thanks for laughing with us! Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.\n")


if __name__ == "__main__":
    main()
