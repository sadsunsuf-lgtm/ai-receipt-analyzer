def categorize_item(item_name):
    categories = {
        'Food': ['milk', 'bread', 'egg', 'apple', 'meat', 'flour'],
        'Personal Care': ['soap', 'shampoo', 'toothpaste'],
        'Utilities': ['bill', 'electric', 'water']
    }
    
    item_lower = item_name.lower()
    for cat, keywords in categories.items():
        if any(key in item_lower for key in keywords):
            return cat
    return "Other"

# Example test
# print(categorize_item("Organic Milk")) # Output: Food

