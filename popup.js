// 팝업이 열리면 chrome.storage에서 저장된 텍스트와 RAG 결과를 가져옴
chrome.storage.local.get(['selectedText', 'selectedTextAnswer'], (result) => {
    const resultHeadDiv = document.getElementById("result_head");
    const resultBodyDiv = document.getElementById("result_body");

    if (result.selectedText) {
        resultHeadDiv.textContent = `검색 문구: ${result.selectedText}`;
        resultBodyDiv.textContent = `검색 결과: ${result.selectedTextAnswer}`;
    } else {
        resultHeadDiv.textContent = "Search Result";
        resultBodyDiv.textContent = "No results available.";
    }
});

