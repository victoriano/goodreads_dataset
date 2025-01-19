from typing import Dict, List, Literal

# Define fiction/non-fiction types
CategoryType = Literal["fiction", "non_fiction"]

# Dictionary mapping categories to their types and descriptions
BOOK_TAXONOMY: Dict[str, Dict[str, str]] = {
    # Fiction Categories
    "literary_fiction": {"type": "fiction", "description": "Literary works focusing on style, character development, and themes"},
    "contemporary_fiction": {"type": "fiction", "description": "Modern-day stories"},
    "lgbtq_fiction": {"type": "fiction", "description": "Fiction centered on LGBTQ+ characters and themes (Also matches: gay, queer, lgbt)"},
    "social_justice_fiction": {"type": "fiction", "description": "Fiction dealing with racism, discrimination, and social justice (Also matches: anti-racism, racial-justice)"},
    "feminist_fiction": {"type": "fiction", "description": "Fiction exploring feminist themes and gender issues"},
    "comedy_fiction": {"type": "fiction", "description": "Humorous fiction and comedic stories (Also matches: humor, humour, funny)"},
    "fantasy": {"type": "fiction", "description": "Fantasy genre with magical or supernatural elements"},
    "high_fantasy": {"type": "fiction", "description": "Fantasy in completely fictional worlds"},
    "urban_fantasy": {"type": "fiction", "description": "Fantasy in modern urban settings"},
    "dark_fantasy": {"type": "fiction", "description": "Blends fantasy with grim, often morally ambiguous themes"},
    "science_fiction": {"type": "fiction", "description": "Fiction based on scientific concepts"},
    "hard_science_fiction": {"type": "fiction", "description": "Focused on scientific accuracy"},
    "soft_science_fiction": {"type": "fiction", "description": "Prioritizes social or emotional aspects over technical details"},
    "time_travel_science_fiction": {"type": "fiction", "description": "Explores the complexities of time travel"},
    "dystopian": {"type": "fiction", "description": "Stories set in oppressive or broken societies"},
    "utopian_fiction": {"type": "fiction", "description": "Idealistic societies in contrast to dystopias"},
    "post_apocalyptic": {"type": "fiction", "description": "Stories set after global catastrophe"},
    "steampunk": {"type": "fiction", "description": "Combines Victorian aesthetics with advanced steam-based technology"},
    "cyberpunk": {"type": "fiction", "description": "Futuristic settings dominated by technology and dystopian themes"},
    "mystery": {"type": "fiction", "description": "Stories involving solving crimes or puzzles"},
    "cozy_mystery": {"type": "fiction", "description": "Light-hearted mysteries"},
    "detective_fiction": {"type": "fiction", "description": "Focuses on professional detectives"},
    "police_procedural": {"type": "fiction", "description": "Centers on law enforcement investigations"},
    "thriller_suspense": {"type": "fiction", "description": "Suspenseful, tension-driven stories"},
    "espionage_thriller": {"type": "fiction", "description": "Spy stories with international intrigue"},
    "legal_thriller": {"type": "fiction", "description": "Drama within the courtroom"},
    "medical_thriller": {"type": "fiction", "description": "Suspense in medical or scientific settings"},
    "horror": {"type": "fiction", "description": "Stories designed to frighten"},
    "gothic_horror": {"type": "fiction", "description": "Horror with gothic elements"},
    "supernatural_horror": {"type": "fiction", "description": "Horror involving supernatural elements"},
    "psychological_horror": {"type": "fiction", "description": "Horror focusing on mental and emotional disturbance"},
    "monster_horror": {"type": "fiction", "description": "Features creatures like zombies, werewolves, or aliens"},
    "romance": {"type": "fiction", "description": "Stories focusing on romantic relationships"},
    "historical_romance": {"type": "fiction", "description": "Romance set in historical periods"},
    "romantic_comedy": {"type": "fiction", "description": "Light-hearted romantic stories"},
    "paranormal_romance": {"type": "fiction", "description": "Romance with supernatural elements"},
    "erotica": {"type": "fiction", "description": "Fiction with explicit sexual content"},
    "historical_fiction": {"type": "fiction", "description": "Fiction set in historical periods"},
    "war_fiction": {"type": "fiction", "description": "Stories about war"},
    "adventure": {"type": "fiction", "description": "Action-packed journey stories"},
    "action": {"type": "fiction", "description": "Fast-paced, action-oriented stories"},
    "western_fiction": {"type": "fiction", "description": "Tales of the American frontier"},
    "young_adult": {"type": "fiction", "description": "Fiction for teenage readers"},
    "coming_of_age": {"type": "fiction", "description": "Stories about growing up"},
    "childrens_literature": {"type": "fiction", "description": "Books for children"},
    "middle_grade": {"type": "fiction", "description": "Books for pre-teen readers"},
    "fairy_tales_folklore": {"type": "fiction", "description": "Traditional, cultural stories often with moral lessons"},
    "mythology_based_fiction": {"type": "fiction", "description": "Retellings or expansions of myths"},
    "graphic_novels": {"type": "fiction", "description": "Stories told through sequential art"},
    "light_novels": {"type": "fiction", "description": "Illustrated, serialized fiction popular in Japan"},
    
    # Non-Fiction Categories
    "lgbtq_studies": {"type": "non_fiction", "description": "Non-fiction works about LGBTQ+ history, culture, and issues (Also matches: queer-studies, gay-studies)"},
    "racial_studies": {"type": "non_fiction", "description": "Books about race, racism, and racial justice (Also matches: anti-racism, race-relations)"},
    "gender_studies": {"type": "non_fiction", "description": "Books about gender, feminism, and women's studies (Also matches: feminism, womens-studies)"},
    "regional_studies": {"type": "non_fiction", "description": "Books about specific countries, regions, or national cultures (Also matches: country-specific, national-studies)"},
    "economics": {"type": "non_fiction", "description": "Books about economic theory, policy, and analysis (Also matches: economy, economic-policy)"},
    "comedy": {"type": "non_fiction", "description": "Non-fiction books about comedy, humor writing, and comedic performance (Also matches: humor-writing, stand-up)"},
    "biography": {"type": "non_fiction", "description": "Life stories written by others"},
    "memoir": {"type": "non_fiction", "description": "Personal life stories"},
    "autobiography": {"type": "non_fiction", "description": "Life stories written by the subject"},
    "self_help": {"type": "non_fiction", "description": "Books for personal improvement"},
    "personal_development": {"type": "non_fiction", "description": "Books for personal growth"},
    "psychology": {"type": "non_fiction", "description": "Books about human behavior and mind"},
    "health_wellness": {"type": "non_fiction", "description": "Books about health and well-being"},
    "fitness_nutrition": {"type": "non_fiction", "description": "Books about exercise and diet"},
    "cookbooks_food": {"type": "non_fiction", "description": "Books about cooking and food"},
    "true_crime": {"type": "non_fiction", "description": "Non-fiction accounts of crimes"},
    "history": {"type": "non_fiction", "description": "Books about historical events"},
    "military_history": {"type": "non_fiction", "description": "Books about military conflicts"},
    "world_history": {"type": "non_fiction", "description": "Global historical events"},
    "cultural_history": {"type": "non_fiction", "description": "Focused on traditions, arts, and societal developments"},
    "ancient_history": {"type": "non_fiction", "description": "History of ancient civilizations"},
    "politics": {"type": "non_fiction", "description": "Books about political systems and events"},
    "political_theory": {"type": "non_fiction", "description": "Theoretical approaches to politics"},
    "philosophy": {"type": "non_fiction", "description": "Books about philosophical thought"},
    "ethics": {"type": "non_fiction", "description": "Books about moral philosophy"},
    "metaphysics": {"type": "non_fiction", "description": "Books about nature of reality"},
    "spirituality": {"type": "non_fiction", "description": "Books about spiritual practices"},
    "religion": {"type": "non_fiction", "description": "Books about religious beliefs"},
    "mythology": {"type": "non_fiction", "description": "Books about myths and legends"},
    "science_nature": {"type": "non_fiction", "description": "Books about scientific topics"},
    "physics": {"type": "non_fiction", "description": "Books about physics"},
    "astronomy": {"type": "non_fiction", "description": "Books about space and celestial objects"},
    "biology": {"type": "non_fiction", "description": "Books about living organisms"},
    "climate_environment": {"type": "non_fiction", "description": "Books about environmental issues"},
    "technology": {"type": "non_fiction", "description": "Books about technological advances"},
    "artificial_intelligence": {"type": "non_fiction", "description": "Books about AI and machine learning"},
    "business_economics": {"type": "non_fiction", "description": "Books about business and economy"},
    "finance_investing": {"type": "non_fiction", "description": "Books about money management"},
    "entrepreneurship": {"type": "non_fiction", "description": "Books about starting businesses"},
    "leadership_management": {"type": "non_fiction", "description": "Books about leading organizations"},
    "education": {"type": "non_fiction", "description": "Books about teaching and learning"},
    "study_guides": {"type": "non_fiction", "description": "Educational study materials"},
    "language_learning": {"type": "non_fiction", "description": "Books for learning languages"},
    "reference": {"type": "non_fiction", "description": "Reference materials"},
    "dictionaries": {"type": "non_fiction", "description": "Word definition books"},
    "encyclopedias": {"type": "non_fiction", "description": "Comprehensive knowledge collections"},
    "how_to_guides": {"type": "non_fiction", "description": "Instructional books"},
    "hobbies_crafts": {"type": "non_fiction", "description": "DIY, knitting, woodworking, etc."},
    "travel_adventure": {"type": "non_fiction", "description": "Books about places and travel"},
    "travelogues": {"type": "non_fiction", "description": "First-hand accounts of journeys"},
    "exploration_expeditions": {"type": "non_fiction", "description": "Books about exploration"},
    "photography_books": {"type": "non_fiction", "description": "Books of/about photography"},
    "art_books": {"type": "non_fiction", "description": "Collections of visual art, paintings, or designs"},
    "music": {"type": "non_fiction", "description": "Books about music and musicians"},
    "pop_culture": {"type": "non_fiction", "description": "Analyses of media, trends, and entertainment"},
    "sports": {"type": "non_fiction", "description": "Books about sports"},
    "esports_gaming": {"type": "non_fiction", "description": "Stories or guides related to video games"},
    "home_garden": {"type": "non_fiction", "description": "Books about home maintenance and gardening"},
    "interior_design": {"type": "non_fiction", "description": "Books about designing spaces"},
    "pets_animals": {"type": "non_fiction", "description": "Books about animals and pet care"},
    "nature_writing": {"type": "non_fiction", "description": "Personal reflections on the natural world"},
    "anthologies": {"type": "non_fiction", "description": "Collections of essays, short stories, or poetry"},
    "essays": {"type": "non_fiction", "description": "Collections of essays"},
    "speech_collections": {"type": "non_fiction", "description": "Collections of speeches"}
}

def get_category_type(category: str) -> CategoryType:
    """Get whether a category is fiction or non-fiction"""
    return BOOK_TAXONOMY[category]["type"]

def get_category_description(category: str) -> str:
    """Get the description of a category"""
    return BOOK_TAXONOMY[category]["description"]

def get_all_categories() -> List[str]:
    """Get a list of all available categories"""
    return list(BOOK_TAXONOMY.keys())

def get_fiction_categories() -> List[str]:
    """Get a list of fiction categories"""
    return [cat for cat, info in BOOK_TAXONOMY.items() if info["type"] == "fiction"]

def get_non_fiction_categories() -> List[str]:
    """Get a list of non-fiction categories"""
    return [cat for cat, info in BOOK_TAXONOMY.items() if info["type"] == "non_fiction"] 