import os
from pathlib import Path

DATA_DIR = Path("backend/RAG/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Templates for realistic real estate content
locations = [
    "DHA Phase 1", "DHA Phase 2", "DHA Phase 3", "DHA Phase 4", "DHA Phase 5",
    "DHA Phase 6", "DHA Phase 7", "DHA Phase 8", "DHA Phase 9 Prism", "DHA Phase 9 Town",
    "Bahria Town Sector A", "Bahria Town Sector B", "Bahria Town Sector C", "Bahria Town Sector D", "Bahria Town Sector E",
    "Bahria Town Sector F", "Bahria Orchard Phase 1", "Bahria Orchard Phase 2", "Bahria Orchard Phase 3", "Bahria Orchard Phase 4",
    "Gulberg Residencia", "Gulberg Greens", "Emaar Canyon Views", "Emaar Panorama", "Eighteen Islamabad",
    "Blue World City", "Capital Smart City", "Nova City", "New City Paradise", "Park View City",
    "Model Town Lahore", "Gulberg Lahore", "Johar Town", "Wapda Town", "Valencia Town",
    "Lake City", "Al Kabir Town", "Kings Town", "Dream Gardens", "Mid City",
    "Gulshan-e-Iqbal Karachi", "Clifton Karachi", "Defense Karachi", "North Nazimabad", "Malir Cantt",
    "Hayatabad Peshawar", "Wapda City Faisalabad", "Canal Garden Multan", "Royal Orchard", "Citi Housing"
]

topics = [
    "Residential Plot Availability", "Commercial Investment Potential", "Development Status Update",
    "Possession and Transfer Process", "Amenity and Infrastructure Overview", "Recent Price Trends",
    "Future Growth Projections", "Security and Gated Features", "Legal Documentation Guide"
]

def generate_docs():
    print(f"Generating 50 documents in {DATA_DIR}...")
    for i in range(50):
        loc = locations[i]
        topic = topics[i % len(topics)]
        filename = f"real_estate_doc_{i+1}.txt"
        
        content = f"""# {loc} - {topic}

This report provides a detailed overview of {loc} focusing on {topic}. 

## Summary
{loc} is currently seeing a significant surge in interest. The {topic} has become a focal point for both local and overseas investors.

## Key Features
- Prime location with easy access to main arteries.
- Fully developed infrastructure including electricity, gas, and water.
- High security standards with 24/7 patrolling.
- Nearby schools, hospitals, and shopping malls.

## {topic} Details
Regarding {topic}, the current market shows that {loc} is outperforming neighboring sectors. Investors are encouraged to complete their legal documentation promptly. Possession is expected to be granted for new blocks by the end of the current fiscal year.

## Contact Information
For more details on plots in {loc}, please contact our authorized dealers or visit the project site office.
"""
        with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)
    
    print("Generation complete!")

if __name__ == "__main__":
    generate_docs()
