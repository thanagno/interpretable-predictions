import pandas as pd
import re
import tomotopy as tp
from nltk.corpus import stopwords

# Define the function to preprocess data
def preprocess_data(data):
    """
    Preprocesses the dataset for further processing and analysis.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Dictionary mapping countries to their continents
    country_to_continent = {
        "Africa": ["Algeria", "Angola", "Burundi", "Cameroon", "Chad", "Congo", "Egypt", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Ghana", "Guinea", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"],
        "Asia": ["Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"],
        "Europe": ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"],
        "North America": ["Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"],
        "South America": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"],
        "Oceania": ["Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"],
        "Antarctica": ["Antarctica"]
    }

    # Add missing mappings
    additional_mappings = {
        "Svalbard & Jan Mayen Islands": "Europe",
        "Guadeloupe": "North America",
        "French Guiana": "South America",
        "Equatorial Guinea": "Africa",
        "Hong Kong": "Asia",
        "Saint Pierre and Miquelon": "North America",
        "Northern Mariana Islands": "Oceania",
        "Western Sahara": "Africa",
        "French Polynesia": "Oceania",
        "American Samoa": "Oceania",
        "United States Minor Outlying Islands": "Oceania",
        "Netherlands Antilles": "North America",
        "Brunei Darussalam": "Asia",
        "Lao People's Democratic Republic": "Asia",
        "Mayotte": "Africa",
        "Korea": "Asia",
        "Libyan Arab Jamahiriya": "Africa",
        "Turks and Caicos Islands": "North America",
        "French Southern Territories": "Antarctica",
        "United States of America": "North America",
        "Guernsey": "Europe",
        "Cook Islands": "Oceania",
        "Puerto Rico": "North America",
        "Bouvet Island (Bouvetoya)": "Antarctica",
        "Cote d'Ivoire": "Africa",
        "United States Virgin Islands": "North America",
        "Faroe Islands": "Europe",
        "Jersey": "Europe",
        "Cayman Islands": "North America",
        "Guam": "Oceania",
        "Sao Tome and Principe": "Africa",
        "Kyrgyz Republic": "Asia",
        "New Caledonia": "Oceania",
        "Macedonia": "Europe",
        "Saint Helena": "Africa",
        "Greenland": "North America",
        "Heard Island and McDonald Islands": "Antarctica",
        "Gibraltar": "Europe",
        "Anguilla": "North America",
        "Russian Federation": "Europe",
        "South Georgia and the South Sandwich Islands": "Antarctica",
        "Christmas Island": "Oceania",
        "Central African Republic": "Africa",
        "Burkina Faso": "Africa",
        "Palestinian Territory": "Asia",
        "Isle of Man": "Europe",
        "Gambia": "Africa",
        "Norfolk Island": "Oceania",
        "Wallis and Futuna": "Oceania",
        "Tokelau": "Oceania",
        "Niue": "Oceania",
        "Syrian Arab Republic": "Asia",
        "Guinea-Bissau": "Africa",
        "Saint Barthelemy": "North America",
        "Reunion": "Africa",
        "Pitcairn Islands": "Oceania",
        "Montserrat": "North America",
        "Falkland Islands (Malvinas)": "South America",
        "Martinique": "North America",
    }

    # Flatten the country-to-continent mapping
    country_continent_map = {**{c: k for k, v in country_to_continent.items() for c in v}, **additional_mappings}

    # Map countries to continents
    data['Continent'] = data['Country'].map(country_continent_map)

    # Check for any unmapped countries
    unmapped_countries = data[data['Continent'].isnull()]['Country'].unique()
    if len(unmapped_countries) > 0:
        print("Remaining Unmapped Countries:", unmapped_countries)
    else:
        print("All countries mapped successfully.")

    # Drop unnecessary columns
    data.drop(columns=['Country', 'City'], inplace=True)

    # print (data.head())

    # Preprocess "Ad Topic Line"
    stopwords_list = stopwords.words('english')

    def tokenizer(doc, sw):
        """Tokenizes and cleans text by removing non-alphabetical characters, stopwords, and short words."""
        return [word for word in [re.sub('[^a-z]', '', x.lower()) for x in doc.strip().split()] if word not in sw and len(word) > 2]

    data['Processed_Ad_Topic_Line'] = data['Ad Topic Line'].apply(lambda x: tokenizer(x, stopwords_list))
    data['Processed_Ad_Topic_Line_Joined'] = data['Processed_Ad_Topic_Line'].apply(lambda x: ' '.join(x))

    # Train LDA model and generate topics
    lda = tp.LDAModel(k=15)  # Number of topics
    for doc in data['Processed_Ad_Topic_Line']:
        lda.add_doc(doc)

    for i in range(0, 500, 10):
        lda.train(10)
        print(f"Iteration: {i}\tLog-likelihood: {lda.ll_per_word}")

    for k in range(lda.k):
        topk_words = [pair[0] for pair in lda.get_topic_words(k, top_n=15)]
        print(f"Topic {k}: {topk_words}\n")

    # Generate topic distributions for each document
    topic_distributions = [doc.get_topic_dist() for doc in lda.docs]
    topic_df = pd.DataFrame(topic_distributions, columns=[f'Topic_{i}' for i in range(lda.k)])

    # Merge topic distributions back into the dataset
    data = pd.concat([data.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)

    # Drop processed columns
    data.drop(columns=['Processed_Ad_Topic_Line', 'Processed_Ad_Topic_Line_Joined'], inplace=True)

    return data
