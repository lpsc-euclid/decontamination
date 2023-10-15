/*--------------------------------------------------------------------------------------------------------------------*/

document.addEventListener('DOMContentLoaded', function() {

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('li.right').forEach(function(item) {

        if(item.querySelector('a[href="py-modindex.html"]'))
        {
            item.style.display = 'none';
        }
    });

    /*----------------------------------------------------------------------------------------------------------------*/

    document.querySelectorAll('.reference .pre').forEach((item) => {

        item.textContent = item.textContent.split('.').pop();
    });

    /*----------------------------------------------------------------------------------------------------------------*/
});

/*--------------------------------------------------------------------------------------------------------------------*/