import json

# Load topics
with open('data/topic_model_results.json', 'r') as f:
    topics = json.load(f)

# Keywords indicating children's content
child_indicators = [
    'student', 'students', 'children', 'kids', 'child', 'grade', 'class',
    'school', 'teacher', 'pupil', 'age', 'years old', 'classroom',
    'elev', 'skole', 'barn',  # Danish/Norwegian
    'estudiante', 'niño', 'alumno', 'escuela',  # Spanish
    'aluno', 'criança', 'escola'  # Portuguese
]


# Keywords indicating self-introduction/presentation texts
self_intro_indicators = [
    # Personal pronouns & identity
    'my name', 'i am', "i'm", 'me llamo', 'mi nombre', 'jeg heter', 'jeg hedder',
    'meu nome', 'jag heter', 'ik ben', 'ich heiße',
    
    # Age
    'years old', 'año', 'años', 'år', 'ano', 'anos', 'age',
    
    # Location/origin
    'i live', 'i come from', 'from', 'vivo en', 'soy de', 'jeg bor', 'mora',
    'vengo de', 'komme fra', 'wohne',
    
    # Family
    'my family', 'my parents', 'my brother', 'my sister', 'mi familia', 
    'familia', 'mor', 'far', 'søster', 'bror', 'padre', 'madre', 'hermano',
    
    # Hobbies/interests
    'i like', 'i love', 'my hobbies', 'me gusta', 'jeg liker', 'gosto de',
    'favorite', 'favorito', 'favorit', 'interests', 'intereses',
    
    # School/grade
    'my school', 'mi escuela', 'min skole', 'minha escola',
    
    # Greetings
    'hello', 'hi', 'hola', 'hej', 'olá', 'salut', 'hallo',
    
    # Personal description
    'about me', 'about myself', 'myself', 'who am i', 'presentation',
    'introducción', 'presentación', 'præsentation'
]

print("Topics potentially about children's writing:\n")
print("=" * 80)

found_any = False
for topic in topics:
    topic_id = topic.get('topic_id', -1)
    keywords = topic.get('keywords', [])
    sample = topic.get('sample', '')
    
    # Check keywords and sample text
    keywords_text = ' '.join(keywords).lower()
    sample_text = sample.lower()
    
    matches = [word for word in self_intro_indicators if word in keywords_text or word in sample_text]
    
    if matches:
        found_any = True
        print(f"\nTopic {topic_id} ({topic.get('num_docs', 0)} docs)")
        print(f"Keywords: {', '.join(keywords[:15])}")
        print(f"Indicators found: {', '.join(set(matches))}")
        print(f"Sample: {sample[:200]}...")
        print("-" * 80)

if not found_any:
    print("\nNo obvious topics about children's writing found in keywords.")
    print("Checking samples more broadly...\n")
    
    for topic in topics:
        sample = topic.get('sample', '').lower()
        if any(indicator in sample for indicator in ['student', 'grade', 'class', 'school', 'age']):
            print(f"Topic {topic.get('topic_id')}: {topic.get('sample')[:150]}...")