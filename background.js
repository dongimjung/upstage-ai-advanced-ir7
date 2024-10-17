console.log("background.js Loaded")
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("on Message~~~")
    if (message.action === "searchRAG") {
        const selectedText = message.text;
        console.log("selectedText : ", selectedText);

        // Flask 서버에 RAG 요청 보내기
        fetch('http://3.36.116.251:5000/rag', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            const selectedTextAnswer = data.answer;
            console.log(" selectedTextAnswer   :::: "+ selectedTextAnswer)

            // 드래그된 텍스트와 RAG 결과를 chrome.storage에 저장
            chrome.storage.local.set({
                selectedText: selectedText,
                selectedTextAnswer: selectedTextAnswer
            }, () => {
                console.log('Text stored:', selectedText);
                console.log('Answer stored:', selectedTextAnswer);

                // 팝업 창 생성
                chrome.windows.create({
                    url: chrome.runtime.getURL("popup.html"),
                    type: "popup",
                    width: 400,
                    height: 300
                });
            });
        })
        .catch(error => {
            console.error("Error fetching RAG result:", error);
        });
    }
});







