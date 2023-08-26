// document.addEventListener("DOMContentLoaded", function () {
//     const folderInput = document.getElementById("folderInput");
//     const uploadFolderButton = document.getElementById("uploadFolderButton");
//     const folderStatus = document.getElementById("folderStatus");

//     uploadFolderButton.addEventListener("click", function () {
//         const folder = folderInput.files[0];
//         if (!folder) {
//             folderStatus.innerText = "No folder selected.";
//             return;
//         }

//         const formData = new FormData();
//         formData.append("folder", folder);

//         fetch("/upload-folder", {
//             method: "POST",
//             body: formData,
//         })
//             .then((response) => response.json())
//             .then((data) => {
//                 if (data.status === "success") {
//                     folderStatus.innerText = "Folder uploaded successfully.";
//                 } else {
//                     folderStatus.innerText = `Error: ${data.message}`;
//                 }
//             })
//             .catch((error) => {
//                 folderStatus.innerText = `Error: ${error.message}`;
//             });
//     });
// });