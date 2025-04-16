import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

def weighted_adoption_speed(row):
    score = 0

    # Faster adoption signals
    if row['PhotoCount'] > 2:
        score += 1
    if row['DescriptionLength'] > 300:
        score += 1
    if row['Sterilized'] == "Yes":
        score += 0.5
    if row['AgeInMonths'] < 12:
        score += 1
    if row['GoodWithChildren'] == "Yes" and row['GoodWithOtherPets'] == "Yes":
        score += 1

    # Slower adoption signals
    if row['Health'] == "Serious Injury":
        score -= 2
    if "Aggressive" in row['Temperament']:
        score -= 1
    if row['AgeInMonths'] > 96:
        score -= 1

    # Map score to adoption speed class
    if score >= 3:
        return 4  # Very fast
    elif score >= 2:
        return 3  # Fast
    elif score >= 1:
        return 2  # Medium
    elif score >= 0:
        return 1  # Slow
    else:
        return 0  # Not adopted

def generate_fake_pet_data(num_records=1000):
    breeds = ['Labrador', 'Poodle', 'German Shepherd', 'Bulldog', 'Mixed', 
              'Beagle', 'Golden Retriever', 'Australian Shepherd', 'Husky']
    sizes = ["Tiny", 'Small', 'Medium', 'Large', "XL"]
    health_status = ['Healthy', 'Minor Injury', 'Serious Injury']
    sterilized = ['Yes', 'No']
    temperament_options = ['Playful', 'Shy', 'Calm', 'Aggressive', 'Friendly', 
                           'Independent', 'Anxious', 'Affectionate']
    yes_no = ['Yes', 'No']
    
    data = []
    for _ in range(num_records):
        breed = random.choice(breeds)
        age = np.random.randint(1, 120)
        size = random.choice(sizes)
        is_sterilized = random.choice(sterilized)
        health = random.choices(health_status, weights=[0.7, 0.2, 0.1])[0]
        photo_count = np.random.randint(0, 6)
        desc_length = np.random.randint(20, 500)
        temperament_combo = ', '.join(random.sample(temperament_options, 3))
        good_with_kids = random.choices(yes_no, weights=[0.8, 0.2])[0]
        good_with_pets = random.choices(yes_no, weights=[0.75, 0.25])[0]

        row = {
            'Breed': breed,
            'AgeInMonths': age,
            'Size': size,
            'Sterilized': is_sterilized,
            'Health': health,
            'Temperament': temperament_combo,
            'GoodWithChildren': good_with_kids,
            'GoodWithOtherPets': good_with_pets,
            'PhotoCount': photo_count,
            'DescriptionLength': desc_length
        }

        row['AdoptionSpeed'] = weighted_adoption_speed(row)
        data.append(row)

    df = pd.DataFrame(data)

    # Save to Downloads
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_path, "realistic_pet_adoption_data.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Data saved to: {file_path}")

# Generate
generate_fake_pet_data(1000)
