"""Spend categories: the labels the model learns.

The source data uses the US government's Product and Service Codes (PSC), a detailed spend taxonomy with
thousands of codes. An accounting team does not need that detail to post an invoice line, so the codes are
grouped into 14 business categories, similar to a company's spend or GL category list.

PSC structure:
- Services start with a letter: R1xx professional support, J0xx equipment maintenance, Z1xx building repair, ...
- Products start with two digits, the Federal Supply Group: 59 electrical components, 66 instruments, ...
- IT products since 2020 start with "7" followed by a letter (7A21, 7B22, ...).
"""

CATEGORIES = [
    "Engineering & professional services",
    "Research & development",
    "Facility operations & management",
    "Construction & building maintenance",
    "Equipment maintenance & repair",
    "IT & telecom services",
    "Environmental services",
    "Utilities, security & housekeeping",
    "Other services",
    "IT hardware & software",
    "Electrical & power equipment",
    "Lab instruments & chemicals",
    "Vehicles & machinery",
    "Other goods",
]
LABEL2ID = {name: i for i, name in enumerate(CATEGORIES)}
ID2LABEL = dict(enumerate(CATEGORIES))

SERVICE_LETTERS = {
    "R": "Engineering & professional services",   # professional, administrative and management support
    "B": "Engineering & professional services",   # special studies and analysis (not R&D)
    "C": "Engineering & professional services",   # architect and engineering services
    "L": "Engineering & professional services",   # technical representative services
    "A": "Research & development",
    "M": "Facility operations & management",       # operation of government-owned facilities
    "Y": "Construction & building maintenance",    # construction of structures and facilities
    "Z": "Construction & building maintenance",    # maintenance, repair, alteration of real property
    "J": "Equipment maintenance & repair",         # maintenance, repair, rebuilding of equipment
    "K": "Equipment maintenance & repair",         # modification of equipment
    "N": "Equipment maintenance & repair",         # installation of equipment
    "H": "Equipment maintenance & repair",         # quality control, testing, inspection
    "D": "IT & telecom services",
    "F": "Environmental services",                 # natural resources, conservation, environmental cleanup
    "S": "Utilities, security & housekeeping",
}
PRODUCT_GROUPS = {
    "70": "IT hardware & software",
    "58": "IT hardware & software",                # communication, detection, radiation equipment
    "59": "Electrical & power equipment",          # electrical and electronic components
    "61": "Electrical & power equipment",          # electric wire, power and distribution equipment
    "62": "Electrical & power equipment",          # lighting fixtures and lamps
    "66": "Lab instruments & chemicals",           # instruments and laboratory equipment
    "67": "Lab instruments & chemicals",           # photographic equipment
    "68": "Lab instruments & chemicals",           # chemicals
    "23": "Vehicles & machinery",                  # motor vehicles
    "24": "Vehicles & machinery",                  # tractors
    "34": "Vehicles & machinery",                  # metalworking machinery
    "35": "Vehicles & machinery",                  # service and trade equipment
    "36": "Vehicles & machinery",                  # special industry machinery
    "38": "Vehicles & machinery",                  # construction, mining, excavating equipment
    "39": "Vehicles & machinery",                  # materials handling equipment
    "41": "Vehicles & machinery",                  # refrigeration and air conditioning equipment
    "43": "Vehicles & machinery",                  # pumps and compressors
    "48": "Vehicles & machinery",                  # valves
}


def psc_to_category(code):
    """Map a PSC code (e.g. 'R425', 'J059', '6640', '7A21') to one of CATEGORIES, or None if empty."""
    if not isinstance(code, str):
        return None
    code = str(code).strip().upper()
    if not code:
        return None
    first = code[0]
    if first.isalpha():
        return SERVICE_LETTERS.get(first, "Other services")
    if first == "7" and len(code) > 1 and code[1].isalpha():
        return "IT hardware & software"
    return PRODUCT_GROUPS.get(code[:2], "Other goods")
