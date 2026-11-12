
from unittest.mock import patch

from retrieval_service import retrieve_context


fake_matches = [
    {
        "id": "test-document-page-1-chunk-0",
        "score": 0.92,
        "metadata": {
            "filename": "leave-policy.pdf",
            "page_number": 1,
            "text": "Employees are entitled to annual leave."
        }
    }
]


with patch(
    "retrieval_service.generate_embedding",
    return_value=[0.1] * 768
) as mock_embedding:

    with patch(
        "retrieval_service.query_vectors",
        return_value=fake_matches
    ) as mock_query:

        result = retrieve_context(
            "What is the annual leave policy?"
        )

        assert result["total_matches"] == 1
        assert "annual leave" in result["context"]
        assert result["sources"][0]["filename"] == "leave-policy.pdf"
        assert result["sources"][0]["page_number"] == 1

        mock_embedding.assert_called_once()
        mock_query.assert_called_once()

        print("Retrieval logic test passed!")
        print(result)



from unittest.mock import patch

from retrieval_service import answer_policy_question


fake_retrieved = {
    "context": (
        "[Source 1]\n"
        "Filename: leave-policy.pdf\n"
        "Page: 1\n"
        "Content: Employees are entitled to annual leave."
    ),
    "sources": [
        {
            "filename": "leave-policy.pdf",
            "page_number": 1,
            "score": 0.92
        }
    ],
    "total_matches": 1
}


with patch(
    "retrieval_service.retrieve_context",
    return_value=fake_retrieved
):

    with patch(
        "retrieval_service.generate_response",
        return_value=(
            "Employees are entitled to annual leave. [Source 1]"
        )
    ) as mock_groq:

        result = answer_policy_question(
            "What is the annual leave policy?"
        )

        assert "annual leave" in result["answer"]
        assert result["sources"][0]["filename"] == "leave-policy.pdf"
        assert result["total_matches"] == 1

        mock_groq.assert_called_once()

        print("Grounded answer test passed!")
        print(result)



from unittest.mock import patch

from retrieval_service import answer_policy_question


empty_retrieved = {
    "context": "",
    "sources": [],
    "total_matches": 0
}


with patch(
    "retrieval_service.retrieve_context",
    return_value=empty_retrieved
):

    with patch(
        "retrieval_service.generate_response"
    ) as mock_groq:

        result = answer_policy_question(
            "What is the maternity leave policy?"
        )

        assert result["total_matches"] == 0
        assert result["sources"] == []

        assert (
            result["answer"]
            == "I could not find relevant information "
               "in the available policy documents."
        )

        mock_groq.assert_not_called()

        print("No-context safety test passed!")
        print(result)
