function showAccountDetails(){
    document.getElementById("accountDetailsModal").style.display = "block";
}

function closeModal() {
    document.getElementById("accountDetailsModal").style.display = "none";
}

function showChatHistory() {
    document.getElementById("chatHistoryModal").style.display = "block";
}

function closeChatModal() {
    document.getElementById("chatHistoryModal").style.display = "none";
}
// Optionally reuse window.onclick handler
window.onclick = function (event) {    
    const accountModal = document.getElementById("accountDetailsModal");
    const chatModal = document.getElementById("chatHistoryModal");
    if (event.target === accountModal) accountModal.style.display = "none";
    if (event.target === chatModal) chatModal.style.display = "none";
};
