document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.delete-chat').forEach(button => {
    button.addEventListener('click', async function() {
      const chatId = this.getAttribute('data-chat-id'); // Get chat ID from button
      if (!chatId) {
        console.error('Chat ID not found on button');
        return;
      }
      
      // Show confirmation popup
      if (confirm('Are you sure you want to delete this chat?')) {
        try {
          // Send DELETE request to server (adjust URL as needed)
          const response = await fetch(`/delete_chat/${chatId}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
          });
          
          if (response.ok) {
            // Remove the chat entry from the page
            this.closest('li').remove(); // Assumes button is inside an <li>
            console.log(`Chat ${chatId} deleted successfully`);
          } else {
            alert('Failed to delete chat');
          }
        } catch (error) {
          console.error('Error deleting chat:', error);
          alert('An error occurred');
        }
      }
    });
  });
});