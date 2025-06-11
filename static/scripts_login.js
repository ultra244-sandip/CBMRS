document.addEventListener('DOMContentLoaded', () => {
  const loginForm = document.getElementById('loginForm');
  const registerForm = document.getElementById('registerForm');
  const toRegister = document.getElementById('toRegister');
  const toLogin = document.getElementById('toLogin');

  // Show login form by default
  loginForm.classList.add('active');

  // Switch to register from login form
  toRegister.addEventListener('click', (e) => {
    e.preventDefault();
    registerForm.classList.add('active');
    loginForm.classList.remove('active');
  });

  // Switch to login from register form
  toLogin.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm.classList.add('active');
    registerForm.classList.remove('active');
  });

  //-------------------------------- Utility Functions --------------------------------//
  function showToast(message, type) {
    const colors = {
      success: "#28a745",
      error: "#dc3545",
      warning: "#ffc107",
      message: "#17a2b8"
    };

    Swal.fire({
      toast: true,
      position: "top",
      icon: type,
      title: message,
      showConfirmButton: false,
      timer: 1500,
      background: colors[type] || "#333", // default dark gray
      color: "#fff",
      customClass: {
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

  function clearFields(form) {
    form.querySelectorAll("input").forEach(input => (input.value = ""));
  }

  function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }

  //-------------------------------- OTP Modal Functions --------------------------------//
  function showModal(modalID) {
    document.getElementById(modalID).style.display = "block";
  }

  function hideModal(modalID) {
    document.getElementById(modalID).style.display = "none";
  }

  // Close the modal when clicking the "×" icon
  const closeModalButton = document.querySelector(".close-modal");
  if (closeModalButton) {
    closeModalButton.addEventListener("click", () => {
      hideModal("otpModal");
    });
  }

  // Optionally, close the modal when clicking outside of its content area.
  window.addEventListener("click", (event) => {
    const otpModal = document.getElementById("otpModal");
    if (event.target === otpModal) {
      hideModal("otpModal");
    }
  });

  //---------------------------- Registration Form Submission ----------------------------//
  registerForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const username = registerForm.querySelector("input[placeholder='Username']").value;
    const email = registerForm.querySelector("input[placeholder='Email Id']").value;
    const password = registerForm.querySelector("input[placeholder='Password']").value;
    const confirmPassword = registerForm.querySelector("input[placeholder='Confirm password']").value;

    // Validate all fields are present
    if (!username || !email || !password || !confirmPassword) {
      showToast("All fields are required!", "error");
      clearFields(registerForm);
      return;
    }

    // Check email format
    if (!isValidEmail(email)) {
      showToast("Invalid email format!", "error");
      registerForm.querySelector("input[placeholder='Email Id']").value = "";
      return;
    }

    // Check if passwords match
    if (password !== confirmPassword) {
      showToast("Passwords do not match!", "error");
      registerForm.querySelector("input[placeholder='Password']").value = "";
      registerForm.querySelector("input[placeholder='Confirm password']").value = "";
      return;
    }

    // All validations passed – send registration details to the server.
    fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include", // include cookies for session if needed
      body: JSON.stringify({ username, email, password })
    })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          showToast(data.error, "error");
        } else {
          // Registration details stored and OTP sent by the backend.
          showToast(data.message, "success");
          // Show the OTP modal so the user can verify the OTP.
          showModal("otpModal");
        }
      })
      .catch(err => showToast("Registration error: " + err.message, "error"));
  });

  //---------------------------- OTP Submission Handling ----------------------------//
  
  const submitOtpButton = document.getElementById("submitBtn");
  if (!submitOtpButton) {
    console.error("Submit OTP button with id 'submitBtn' not found. Check your HTML.");
  } else {
    submitOtpButton.addEventListener("click", (event) => {
      event.preventDefault();
      const otpInputElem = document.getElementById("otpInput");
      if (!otpInputElem) {
        console.error("OTP input element with id 'otpInput' not found.");
        return;
      }
      const otp = otpInputElem.value.trim();

      // Check if an OTP is provided
      if (!otp) {
        showToast("Please enter the OTP!", "error");
        return;
      }

      // Send the OTP to the backend for verification.
      fetch("/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ otp: otp })
      })
        .then(response => {
          return response.json();
        })
        .then(data => {
          hideModal("otpModal");
          if (data.error) {
            showToast(data.error, "error");
          } else {
            showToast(data.message || "Registration complete!", "success");
            otpInputElem.value = "";
            setTimeout(() => window.location.href = "/login", 1500);
          }
        })
        .catch(error => {
          console.error("Error during OTP verification:", error);
          hideModal("otpModal");
          showToast("OTP Verification error: " + error.message, "error");
        });
    });
  }

  //---------------------------- Login Form Submission ----------------------------//
  loginForm.addEventListener('submit', (event) => {
    event.preventDefault();

    const username = loginForm.querySelector("input[placeholder='Username']").value;
    const password = loginForm.querySelector("input[placeholder='Password']").value;

    if (!username || !password) {
      showToast("Username and password are required!", "error");
      clearFields(loginForm);
      return;
    }

    fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    })
      .then(response => response.json())
      .then(data => {
        if (data.message === "Login successful") {
          showToast(data.message, "success");
          clearFields(loginForm);
          setTimeout(() => window.location.href = "/", 1500);
        } else {
          showToast(data.message, "error");
          clearFields(loginForm);
        }
      })
      .catch(error => console.error("Login error:", error));
  });
});