import json


# school_id
SCHOOL_MAPPING_CONTEXT = json.dumps(
    {
        "1": {
            "name": "National University of Singapore",
            "aliases": ["NUS", "nus", "National University of Singapore", "National University"]
        },
        "2": {
            "name": "Nanyang Technological University",
            "aliases": ["NTU", "ntu", "Nanyang Technological University", "Nanyang Tech"]
        },
        "3": {
            "name": "Singapore Management University",
            "aliases": ["SMU", "smu", "Singapore Management University", "Management University"]
        },
        "4": {
            "name": "Singapore University of Technology and Design",
            "aliases": ["SUTD", "sutd", "Singapore University of Technology and Design"]
        },
        "5": {
            "name": "Singapore Institute of Technology",
            "aliases": ["SIT", "sit", "Singapore Institute of Technology"]
        },
        "6": {
            "name": "Singapore University of Social Sciences",
            "aliases": ["SUSS", "suss", "Singapore University of Social Sciences", "Social Sciences University"]
        }
    },
    indent=2
)


# target_district_id
DISTRICT_MAPPING_CONTEXT = json.dumps(
    {
        "1": {"name": "Bukit Merah East", "aliases": ["Bukit Merah East", "bukit merah east"]},
        "2": {"name": "Marina Bay", "aliases": ["Marina Bay", "marina bay"]},
        "3": {"name": "Rochor", "aliases": ["Rochor", "rochor"]},
        "4": {"name": "Bukit Merah West", "aliases": ["Bukit Merah West", "bukit merah west"]},
        "5": {"name": "Clementi", "aliases": ["Clementi", "clementi"]},
        "6": {"name": "Jurong East", "aliases": ["Jurong East", "jurong east"]},
        "7": {"name": "Queenstown", "aliases": ["Queenstown", "queenstown"]},
        "8": {"name": "Bishan", "aliases": ["Bishan", "bishan"]},
        "9": {"name": "Bukit Timah", "aliases": ["Bukit Timah", "bukit timah"]},
        "10": {"name": "Kampong Java", "aliases": ["Kampong Java", "kampong java"]},
        "11": {"name": "Orchard", "aliases": ["Orchard", "orchard", "Orchard Road"]},
        "12": {"name": "Toa Payoh", "aliases": ["Toa Payoh", "toa payoh"]},
        "13": {"name": "Ang Mo Kio North", "aliases": ["Ang Mo Kio North", "ang mo kio north", "AMK North"]},
        "14": {"name": "Ang Mo Kio South", "aliases": ["Ang Mo Kio South", "ang mo kio south", "AMK South", "AMK"]},
        "15": {"name": "Hougang", "aliases": ["Hougang", "hougang"]},
        "16": {"name": "Punggol", "aliases": ["Punggol", "punggol"]},
        "17": {"name": "Sengkang", "aliases": ["Sengkang", "sengkang"]},
        "18": {"name": "Serangoon", "aliases": ["Serangoon", "serangoon"]},
        "19": {"name": "Woodleigh", "aliases": ["Woodleigh", "woodleigh"]},
        "20": {"name": "Bedok", "aliases": ["Bedok", "bedok"]},
        "21": {"name": "Changi", "aliases": ["Changi", "changi"]},
        "22": {"name": "Geylang", "aliases": ["Geylang", "geylang"]},
        "23": {"name": "Marine Parade", "aliases": ["Marine Parade", "marine parade"]},
        "24": {"name": "Pasir Ris", "aliases": ["Pasir Ris", "pasir ris"]},
        "25": {"name": "Tampines", "aliases": ["Tampines", "tampines"]},
        "26": {"name": "Bukit Batok", "aliases": ["Bukit Batok", "bukit batok"]},
        "27": {"name": "Bukit Panjang", "aliases": ["Bukit Panjang", "bukit panjang"]},
        "28": {"name": "Choa Chu Kang", "aliases": ["Choa Chu Kang", "choa chu kang", "CCK"]},
        "29": {"name": "Jurong West", "aliases": ["Jurong West", "jurong west"]},
        "30": {"name": "Nanyang", "aliases": ["Nanyang", "nanyang"]},
        "31": {"name": "Woodlands East", "aliases": ["Woodlands East", "woodlands east"]},
        "32": {"name": "Woodlands West", "aliases": ["Woodlands West", "woodlands west", "Woodlands"]},
        "33": {"name": "Yishun North", "aliases": ["Yishun North", "yishun north"]},
        "34": {"name": "Yishun South", "aliases": ["Yishun South", "yishun south", "Yishun"]},
        "35": {"name": "Sembawang", "aliases": ["Sembawang", "sembawang"]},
        "36": {"name": "Checkpoint", "aliases": ["Checkpoint", "checkpoint", "Woodlands Checkpoint", "Tuas Checkpoint"]}
    },
    indent=2
)


# flat_type_preference
FLAT_TYPE_CONTEXT = json.dumps(
    {
        "1": {
            "name": "HDB",
            "aliases": ["HDB", "hdb", "HDB flat", "flat", "public housing"]
        },
        "2": {
            "name": "Condo",
            "aliases": ["Condo", "condo", "condominium", "condos", "private condo"]
        },
        "3": {
            "name": "Landed",
            "aliases": ["Landed", "landed", "landed property", "house", "detached house", "semi-detached", "terrace house"]
        },
        "4": {
            "name": "Apartment",
            "aliases": ["Apartment", "apartment", "apartments", "private apartment", "walk-up apartment"]
        },
        "5": {
            "name": "Executive Condo",
            "aliases": ["Executive Condo", "executive condo", "EC", "ec"]
        }
    },
    indent=2
)
