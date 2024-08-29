const generateButton = document.getElementById('generate-button');
const image = document.getElementById('image');

generateButton.addEventListener('click', () => {
    fetch('/rest/generate')
        .then((response) => {
            response.blob().then((blob) => {
                image.src = window.URL.createObjectURL(blob);
            });
        })
        .catch(console.error);
});