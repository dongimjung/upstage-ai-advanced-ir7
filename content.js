console.log("Content script loaded");

let selectedText = "";

// 아이콘 생성
const searchIcon = document.createElement("div");
searchIcon.style.position = "absolute";
searchIcon.style.display = "none";
searchIcon.style.zIndex = "9999";
searchIcon.style.width = "24px";
searchIcon.style.height = "24px";
searchIcon.style.borderRadius = "50%";
searchIcon.style.backgroundImage = "url('https://demandsa.blob.core.windows.net/inmind/meditation/img/thumb/23_07_10.png')";
searchIcon.style.backgroundSize = "contain";  // 아이콘 크기 맞춤
searchIcon.style.border = "2px solid pink";
searchIcon.style.cursor = "pointer";
searchIcon.title = "RAG 검색";
searchIcon.style.pointerEvents = "auto";  // 클릭 가능한 상태로 변경

// 문서에 아이콘 추가
document.body.appendChild(searchIcon);

// 텍스트 선택 처리
document.addEventListener("mouseup", function (event) {
    try {
        const selection = window.getSelection();
        if (selection) {
            let selected = selection.toString().trim();
            if (selected.length > 0) {
                selectedText = selected;  // 선택된 텍스트 저장
                console.log("Selected text: ", selectedText);

                // 마우스 커서 근처에 아이콘 위치
                searchIcon.style.top = `${event.pageY - 20}px`;
                searchIcon.style.left = `${event.pageX }px`;  // 약간 오른쪽에
                searchIcon.style.display = "block";  // 아이콘 표시

            } else {
                searchIcon.style.display = "none";  // 텍스트가 없으면 아이콘 숨기기
            }
        }
    } catch (error) {
        console.error("Error during text selection: ", error);
    }
});

// 아이콘을 클릭했을 때 백그라운드 스크립트로 메시지 전송
searchIcon.addEventListener("click", function (event) {
    event.stopPropagation();  // 부모 요소로 이벤트 전파 중단
    console.log("aaaaa");
    
    if (selectedText.length > 0) {
        console.log("Selected text (from saved): ", selectedText);
        
        // 백그라운드 스크립트로 메시지 전송
        chrome.runtime.sendMessage({ action: "searchRAG", text: selectedText }, (response) => {
            console.log("Message sent to background.js, response:", response);
        });

        // 드래그된 텍스트 취소
        window.getSelection().removeAllRanges();

        // 아이콘 숨기기
        searchIcon.style.display = "none";
    } else {
        console.error("No text selected to send to background.js");
    }
});

// 클릭한 후 텍스트가 없어지면 아이콘 숨기기 (아이콘 클릭 제외)
document.addEventListener("mousedown", function (event) {
    // 아이콘 클릭이 아닌 경우에만 숨기기
    if (event.target !== searchIcon) {
        searchIcon.style.display = "none";
    }
});

