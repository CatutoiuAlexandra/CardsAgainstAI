function createCardElement(cardObject) { //creare card bootstrap
    const h5 = document.createElement('h5');
    h5.className = 'card-title';
    h5.textContent = cardObject.text;

    const a = document.createElement('button');
    a.className = 'btn btn-outline-secondary';
    a.textContent = 'Selectează';

    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    cardBody.style.textAlign = 'center';

    const cardElement = document.createElement('div');
    cardElement.className = 'card';
    cardElement.style.backgroundColor = 'white';
    cardElement.disabled = false;
    cardElement.id = cardObject.id;

    cardBody.appendChild(h5);
    cardBody.appendChild(a);
    cardElement.appendChild(cardBody);

    return cardElement;
}

function createBadgeElementForCard(index) { // creare badge cu index
    let span = document.createElement('span');
    span.className = 'position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger';
    span.textContent = index;
    return span;
}

function makeCardSelected(cardElem) { //marcheaza cardul ca selectat
    cardElem.classList.add(StatusAction.SELECTED); // adaugam element in class
    cardElem.style.backgroundColor = 'green';
}

function makeCardUnselected(cardElem) {
    cardElem.classList.remove(StatusAction.SELECTED); //stergem elementul din class
    cardElem.style.backgroundColor = 'white';
}


  