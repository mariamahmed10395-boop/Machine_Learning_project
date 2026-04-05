# Feature options (from the original census dataset)
WORKCLASS_OPTS = [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
]

EDUCATION_OPTS = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', 
    '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 
    'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
]

MARITAL_OPTS = [
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
    'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
]

OCCUPATION_OPTS = [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
    'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
]

RELATIONSHIP_OPTS = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
RACE_OPTS = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
SEX_OPTS = ['Male', 'Female']

COUNTRY_OPTS = [
    'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
    'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
    'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
    'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
    'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
    'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
    'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'
]

# Color palette (Dark Navy + Teal theme)
COLORS = {
    'bg': '#062C22',          
    'surface': '#0A3D31',     
    'card': '#114D40',     
    'border': '#D4AF37', 
    'accent': '#F1C40F',      
    'accent_light': '#FDE68A',
    'green': '#2ECC71',
    'red': '#E74C3C',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'text': '#ECF0F1',
    'text_secondary': '#BDC3C7',
    'white': '#FFFFFF'
}

# Default feature values
DEFAULT_FEATURES = {
    'age': 38,
    'education-num': 10,
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'workclass': 'Private',
    'education_level': 'Bachelors',
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'native-country': 'United-States'
}

TYPICAL_DONOR = {
    'age': 45,
    'education-num': 13,
    'capital-gain': 7688,
    'capital-loss': 0,
    'hours-per-week': 50,
    'workclass': 'Private',
    'education_level': 'Bachelors',
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'native-country': 'United-States'
}
