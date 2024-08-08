
const myImage = document.querySelector("img");

myImage.onclick = () => {
  const mySrc = myImage.getAttribute("src");
  if (mySrc === "Images/funy.gif") {
    myImage.setAttribute("src", "Images/monkey.avif");
  } else {
    myImage.setAttribute("src", "Images/funy.gif");
  }
};





let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");

function setUserName() {
    const myName = prompt("Please enter your name.");
    if (!myName) {
      setUserName();
    } else {
      localStorage.setItem("name", myName);
      myHeading.textContent = `Hello, ${myName} This is my site`;
    }
  }
  
  

  if (!localStorage.getItem("name")) {
    setUserName();
  } else {
    const storedName = localStorage.getItem("name");
    myHeading.textContent = `Hello, ${storedName} This is my site`;
  }

  
  myButton.onclick = () => {
    setUserName();
  };

