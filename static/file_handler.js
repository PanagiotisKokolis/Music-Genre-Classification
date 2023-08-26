document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("drop");
    const uploadButton = document.getElementById("uploadButton");
    const fileStatus = document.getElementById("uploadStatus");
    const fileList = document.querySelector(".file-list");

    function handleFileSelect(evt) {
        evt.preventDefault();
        const files = evt.target.files || evt.dataTransfer.files;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const listItem = document.createElement("div");
            listItem.className = "file-list-item";
            listItem.innerHTML = `<span>${file.name}</span>`;

            fileList.appendChild(listItem);
        }
    }

    function handleFileDrop(evt) {
        evt.preventDefault();
        fileInput.files = evt.dataTransfer.files;
    }

    function handleUpload() {
        const formData = new FormData();
        const files = fileInput.files;

        for (let i = 0; i < files.length; i++) {
            formData.append("file", files[i]);
        }

        fetch("/upload", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "success") {
                    fileStatus.innerText = `File ${data.filename} uploaded successfully: ${data.message}`;
                    fileStatus.classList.add("success");
                } else {
                    fileStatus.innerText = `Error: ${data.message}`;
                    fileStatus.classList.add("error");
                }
                fileStatus.classList.remove("hidden");
            })
            .catch((error) => {
                fileStatus.innerText = `Error: ${error.message}`;
                fileStatus.classList.add("error");
                fileStatus.classList.remove("hidden");
            });
    }

    // Event listeners
    fileInput.addEventListener("change", handleFileSelect);
    dropZone.addEventListener("dragover", (evt) => evt.preventDefault());
    dropZone.addEventListener("drop", handleFileDrop);
    uploadButton.addEventListener("click", handleUpload);
});
