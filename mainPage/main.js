// Variabile globale
let currentMaxSpaces = 0;
let currentSelectedCards = 0;
let aiCandidateWhiteCards = [];

const firstPunchline = 1;
const secondPunchline = 2;
let firstPunchlineUsed = false;
let secondPunchlineUsed = false;

// Inițializarea paginii
(async () => {
  await displayBlackCard();
  await displayWhiteCards();
  handleCardSelection();
  document.getElementById('submit-button').addEventListener('click', (event) => {
    submitAnswer(event.target);
  });
})();

async function displayBlackCard() {
  const cardObject = await fetchBlackCard();
  const blackCardElement = createCardElement(cardObject, 'black');
  const blackCardContainer = document.getElementById('black-card');
  blackCardContainer.innerHTML = "";
  blackCardContainer.appendChild(blackCardElement);
  currentMaxSpaces = cardObject.spaces;
}

async function displayWhiteCards() {
  const humanCardsArea = document.getElementById('white-cards-area-human');
  humanCardsArea.innerHTML = "";
  const whiteCardsArrayHuman = await fetchWhiteCards();
  whiteCardsArrayHuman.forEach(card => {
    const cardElement = createCardElement(card);
    humanCardsArea.appendChild(cardElement);
  });
  aiCandidateWhiteCards = whiteCardsArrayHuman;
}

function createCardElement(cardObject, cardType = 'white', isAI = false) {
  // Titlul cardului
  const h5 = document.createElement('h5');
  h5.className = "card-title";
  h5.textContent = (cardType === "black") ? cardObject.sentence : cardObject.text;
  
  // Corpul cardului
  const cardBody = document.createElement('div');
  cardBody.className = "card-body";
  cardBody.style.textAlign = "center";
  cardBody.appendChild(h5);
  
  // Containerul intern
  const cardInner = document.createElement('div');
  cardInner.className = "card-inner";
  cardInner.appendChild(cardBody);
  
  // Containerul extern 
  const cardElement = document.createElement('div');
  cardElement.className = "card";
  if (cardType === "black") {
    cardElement.classList.add("black-card");
  } else {
    cardElement.classList.add("white-card");
  }
  cardElement.id = cardObject.id;
  if (isAI) {
    cardElement.classList.add("ai", "disabled");
  }
  cardElement.appendChild(cardInner);
  
  attachRotationEffect(cardElement);
  return cardElement;
}

function attachRotationEffect(cardElement) {
  const cardInner = cardElement.querySelector('.card-inner');
  const maxAngle = 10;
  
  cardElement.addEventListener('mousemove', (e) => {
    const rect = cardElement.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const diffX = (x - centerX) / centerX;
    const diffY = (y - centerY) / centerY;
    const rotateY = diffX * maxAngle;
    const rotateX = -diffY * maxAngle;
    cardInner.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
  });
  
  cardElement.addEventListener('mouseleave', () => {
    cardInner.style.transform = 'rotateX(0deg) rotateY(0deg)';
  });
}

async function handleCardSelection() {
  const whiteCardsArea = document.getElementById('white-cards-area');
  whiteCardsArea.addEventListener('click', (e) => {
    const cardElem = getCardParentElementFromClickAnywhereInCardContainer(e);
    if (!cardElem || cardElem.classList.contains(StatusAction.DISABLED)) return;
    
    if (!cardElem.classList.contains(StatusAction.SELECTED) && currentSelectedCards < currentMaxSpaces) {
      makeCardSelected(cardElem);
      currentSelectedCards++;
      if (currentMaxSpaces === 2) addBadgeElementForCard(cardElem);
      if (currentSelectedCards === currentMaxSpaces) changeDisableStatusForUnusedCards(StatusAction.DISABLED);
    } else if (cardElem.classList.contains(StatusAction.SELECTED)) {
      if (currentSelectedCards === currentMaxSpaces) changeDisableStatusForUnusedCards(StatusAction.ENABLED);
      if (currentMaxSpaces === 2) removeBadgeElementForCard(cardElem);
      currentSelectedCards--;
      makeCardUnselected(cardElem);
    }
    checkValiditySubmitButton();
  });
}

function addBadgeElementForCard(cardElem) {
  if (!firstPunchlineUsed) {
    const badge = createBadgeElementForCard(firstPunchline);
    firstPunchlineUsed = true;
    cardElem.appendChild(badge);
    cardElem.dataset.indexSentence = `${firstPunchline}`;
  } else if (!secondPunchlineUsed) {
    const badge = createBadgeElementForCard(secondPunchline);
    secondPunchlineUsed = true;
    cardElem.appendChild(badge);
    cardElem.dataset.indexSentence = `${secondPunchline}`;
  }
}

function removeBadgeElementForCard(cardElem) {
  const spanChildArray = cardElem.getElementsByTagName('span');
  let spanElem = (spanChildArray.length === 1) ? spanChildArray[0] : null;
  if (spanElem) {
    if (+cardElem.dataset.indexSentence === firstPunchline && firstPunchlineUsed) {
      cardElem.removeAttribute('data-indexSentence');
      cardElem.removeChild(spanElem);
      firstPunchlineUsed = false;
    } else if (+cardElem.dataset.indexSentence === secondPunchline && secondPunchlineUsed) {
      cardElem.removeAttribute('data-indexSentence');
      cardElem.removeChild(spanElem);
      secondPunchlineUsed = false;
    }
  }
}

function getCardParentElementFromClickAnywhereInCardContainer(event) {
  return event.target.closest('.card');
}

function changeDisableStatusForUnusedCards(status) {
  const allCards = document.getElementsByClassName('card');
  Array.from(allCards).forEach(card => {
    if (!card.classList.contains(StatusAction.SELECTED)) {
      if (status === StatusAction.DISABLED) {
        card.classList.add(StatusAction.DISABLED);
        card.style.cursor = "normal";
        card.style.opacity = "0.5";
      } else {
        card.classList.remove(StatusAction.DISABLED);
        card.style.cursor = "pointer";
        card.style.opacity = "1";
      }
    }
  });
}

function resetCards() {
  const allCards = document.getElementsByClassName('card');
  Array.from(allCards).forEach(card => {
    card.classList.remove(StatusAction.DISABLED);
    card.style.cursor = "pointer";
    card.style.opacity = "1";
  });
}

function checkValiditySubmitButton() {
  const submitButton = document.getElementById('submit-button');
  if (currentSelectedCards === currentMaxSpaces && submitButton.disabled) {
    submitButton.disabled = false;
  } else if (currentSelectedCards !== currentMaxSpaces && !submitButton.disabled) {
    submitButton.disabled = true;
  }
}

async function submitAnswer(submitButton) {
  if (currentSelectedCards === currentMaxSpaces && !submitButton.disabled) {
    const playerAnswer = getSelectedCardsInfo();
    const candidateWhiteCardsText = aiCandidateWhiteCards.map(card => card.text);
    const aiData = await submitAIAnswer(pickedBlackCard.sentence, candidateWhiteCardsText);
    const aiAnswer = aiData ? aiData.funniest : "No answer available";
    resetCards();
    displayFinalScreen(pickedBlackCard.sentence, playerAnswer, aiAnswer);
  }
}

function getSelectedCardsInfo() {
  const allCards = document.getElementsByClassName('card');
  let response = '';
  Array.from(allCards).forEach(card => {
    if (card.classList.contains(StatusAction.SELECTED)) {
      if (currentMaxSpaces === 1) {
        response = card.firstElementChild.firstElementChild.innerText;
      } else {
        response += card.dataset.indexSentence + ": " + card.firstElementChild.firstElementChild.innerText + "\n";
      }
    }
  });
  return response;
}

function displayFinalScreen(blackCardText, playerAnswer, aiAnswer) {
  document.getElementById('game-area').style.display = "none";
  
  const finalBlackCardContainer = document.getElementById('final-black-card');
  finalBlackCardContainer.innerHTML = "";
  const blackCardObj = { id: 'finalBlackCard', sentence: blackCardText, spaces: 1 };
  const finalBlackCard = createCardElement(blackCardObj, 'black');
  finalBlackCardContainer.appendChild(finalBlackCard);
  
  const finalPlayerContainer = document.getElementById('final-player-answer');
  finalPlayerContainer.innerHTML = "";
  const playerCardObj = { id: 'finalPlayerCard', text: playerAnswer };
  const finalPlayerCard = createCardElement(playerCardObj, 'white');
  finalPlayerContainer.appendChild(finalPlayerCard);
  
  const finalAIContainer = document.getElementById('final-ai-answer');
  finalAIContainer.innerHTML = "";
  const aiCardObj = { id: 'finalAICard', text: aiAnswer };
  const finalAICard = createCardElement(aiCardObj, 'white');
  finalAIContainer.appendChild(finalAICard);
  
  document.getElementById('final-screen').style.display = "block";
}

function restartGame() {
  window.location.reload();
}

