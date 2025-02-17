/*--------------------------------------------------------------------------------------------------------------------*/

document.addEventListener('DOMContentLoaded', function() {

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('img[crossorigin="anonymous"]').forEach((item) => {

        const url = new URL(item.src);

        url.searchParams.set('_', `${Math.floor(Date.now() / (1000 * 60))}`);

         item.src = 'https://corsproxy.io/?' + encodeURIComponent(url.toString());
    });

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('#right-sidebar .pre').forEach((item) => {

        item.textContent = item.textContent.split('.').pop();
    });

    /*----------------------------------------------------------------------------------------------------------------*/
});

/*--------------------------------------------------------------------------------------------------------------------*/
