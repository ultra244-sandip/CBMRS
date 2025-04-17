document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const toRegister = document.getElementById('toRegister');
    const toLogin = document.getElementById('toLogin');
    
    // Show login form by default
    loginForm.classList.add('active');
    
    // Link to register from login
    toRegister.addEventListener('click', (e) => {
        e.preventDefault();
        registerForm.classList.add('active');
        loginForm.classList.remove('active');
    });

    // Link to login from register
    toLogin.addEventListener('click', (e) => {
        e.preventDefault();
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
    });

    //---------------------------New Code---------------------------//
    function showToast(message, type) {
        const colors = {
            success : "#28a745",
            error : "#dc3545",
            warning : "#ffc107",
            info : "#17a2b8"
        };
    
        Swal.fire({
            toast: true,
            position: "top",
            icon: type,
            title: message,
            showConfirmButton: false,
            timer: 1500,
            background: colors[type] || "#333",  // Default dark gray
            color: "#fff",
            customClass:{
                popup: "colored-toast"
            },
            didOpen: () => {
                document.body.style.overflow = "hidden";
            },
            didClose: () => {
                document.body.style.overflow = "";
            }
        });
    }

    function clearFields(form){
        form.querySelectorAll("input").forEach(input => input.value = "");
    }

    function isValidEmail(email){
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }
    
    registerForm.addEventListener('submit', (event)=>{
        event.preventDefault();

        const username = registerForm.querySelector("input[placeholder='Username']").value;
        const email = registerForm.querySelector("input[placeholder='Email Id']").value;
        const password = registerForm.querySelector("input[placeholder='Password']").value;
        const confirmPassword = registerForm.querySelector("input[placeholder='Confirm password']").value;

        if (!username || !email || !password || !confirmPassword) {
            showToast("All fields are required!", "error");
            clearFields(registerForm);
            return;
        }

        if (!isValidEmail(email)) {
            showToast("Invalid email format!", "error");
            registerForm.querySelector("input[placeholder='Email Id']").value = "";
            return;
        }

        if(password != confirmPassword){
            showToast("Password do not match!","error");
            registerForm.querySelector("input[placeholder='Password']").value = "";
            registerForm.querySelector("input[placeholder='Confirm password']").value = "";
            return;
        }
        
        //Handle registration
        fetch("/register",{
            method: "POST",
            headers: {"Content-type" : "application/json"},
            body: JSON.stringify({username, email, password})
        })
        .then(response => response.json())
        .then(data => {
            if(data.message.includes("success")){
                showToast(data.message, "success");
                clearFields(registerForm);
                setTimeout(() => window.location.href = "/loging", 1500);
            }
            else{
                showToast(data.message, "error");
            }
        })
        .catch(error => console.error("Error:", error));
    });

    loginForm.addEventListener('submit', (event) =>{
        event.preventDefault();
        
        const username = loginForm.querySelector("input[placeholder='Username']").value;
        const password = loginForm.querySelector("input[placeholder='Password']").value;
        
        if (!username || !password) {
            showToast("User name and password are required!","error");
            clearFields(loginForm);
            return;
        }

        //Handle login
        fetch("/login",{
            method: "POST",
            headers: {"Content-type" : "application/json"},
            body: JSON.stringify({username, password})
        })
        .then(response => response.json())
        .then(data => {
            if (data.message === "Login successful") {
                showToast(data.message, "success");
                clearFields(loginForm);
                setTimeout(() => window.location.href = "/", 1500);
            }
            else{
                showToast(data.message,"error");
                clearFields(loginForm);
            }
        })
        .catch(error => console.error("Error:", error));
    });
});

