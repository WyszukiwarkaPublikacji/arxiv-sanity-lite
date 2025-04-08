from aslite.db import get_papers_db

# Example data
example_data = {
    "1234.56789v1": {
        "arxiv_id": "1234.56789v1",
        "links": [{"title": "pdf", "href": "https://arxiv.org/pdf/2504.03649"}]
    },
    "1234.56789v2": {
        "arxiv_id": "1234.56789v2",
        "links": [{"title": "pdf", "href": "https://arxiv.org/pdf/2504.03699"}]
    },
    "1234.56780": {
        "arxiv_id": "1234.56780",
        "links": [{"title": "pdf", "href": "https://arxiv.org/pdf/2504.03720"}]
    }
}

# Load papers.db using the correct format
papers_db = get_papers_db(flag='c')  # 'c' = read/write/create

# Insert data
for key, value in example_data.items():
    papers_db[key] = value

# Save changes
papers_db.commit()
print("Successfully populated papers.db")
