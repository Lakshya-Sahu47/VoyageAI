// Dynamic frontend logic for VoyageAI Travel Recommendation Engine

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Date Selectors (default: tomorrow to 4 days from now)
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const futureDate = new Date(tomorrow);
    futureDate.setDate(futureDate.getDate() + 3);

    startDateInput.value = formatDate(tomorrow);
    endDateInput.value = formatDate(futureDate);

    // 2. Interactive Star Ratings Logic
    const categoryRatings = {
        'private_&_custom_tours': 5,
        'tours_&_sightseeing': 5,
        'cultural_&_theme_tours': 5,
        'sightseeing_tickets_&_passes': 5,
        'multi-day_&_extended_tours': 5
    };

    const starsContainers = document.querySelectorAll('.stars');
    starsContainers.forEach(container => {
        const category = container.getAttribute('data-category');
        const stars = container.querySelectorAll('i');
        
        // Initialize default rating stars (all 5 active)
        updateStars(stars, 5);

        stars.forEach(star => {
            star.addEventListener('click', () => {
                const val = parseInt(star.getAttribute('data-value'));
                categoryRatings[category] = val;
                updateStars(stars, val);
            });
        });
    });

    function updateStars(stars, rating) {
        stars.forEach(star => {
            const val = parseInt(star.getAttribute('data-value'));
            if (val <= rating) {
                star.classList.add('active');
            } else {
                star.classList.remove('active');
            }
        });
    }

    // 3. Preferred Hotel Amenities Chip Toggles
    const chips = document.querySelectorAll('.chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            chip.classList.toggle('active');
        });
    });

    function getSelectedAmenities() {
        const activeChips = document.querySelectorAll('.chip.active');
        return Array.from(activeChips).map(chip => chip.getAttribute('data-amenity'));
    }

    // 4. Tab Switcher Logic
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            
            // Toggle active buttons
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active contents
            tabContents.forEach(content => {
                if (content.id === `tab-${tabName}`) {
                    content.classList.add('active');
                    content.style.display = 'block';
                } else {
                    content.classList.remove('active');
                    content.style.display = 'none';
                }
            });
        });
    });

    // 5. Submit Form & Call Recommendations Backend API
    const form = document.getElementById('recommendation-form');
    const loader = document.getElementById('results-loader');
    const emptyState = document.getElementById('results-empty');
    const tabItinerary = document.getElementById('tab-itinerary');
    const tabHotels = document.getElementById('tab-hotels');
    const itineraryContainer = document.getElementById('itinerary-days-container');
    const hotelsContainer = document.getElementById('hotels-container');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        loader.style.display = 'flex';
        emptyState.style.display = 'none';
        tabItinerary.style.display = 'none';
        tabHotels.style.display = 'none';
        
        // Assemble request body
        const reqData = {
            username: document.getElementById('username').value,
            province: document.getElementById('destination').value,
            budget_low: document.getElementById('budget-low').value,
            budget_high: document.getElementById('budget-high').value,
            start_date: startDateInput.value,
            end_date: endDateInput.value,
            category_ratings: categoryRatings,
            amenities_pref: getSelectedAmenities()
        };

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reqData)
            });

            const result = await response.json();
            
            // Hide loading state
            loader.style.display = 'none';

            if (result.success) {
                // Populate results
                renderItinerary(result.attractions);
                renderHotels(result.hotels);
                
                // Show tab contents (default: itinerary active)
                const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
                if (activeTab === 'itinerary') {
                    tabItinerary.style.display = 'block';
                } else {
                    tabHotels.style.display = 'block';
                }
            } else {
                showError(result.error || 'Server calculation failed.');
            }

        } catch (error) {
            loader.style.display = 'none';
            showError('Unable to connect to the backend server.');
            console.error('Fetch error:', error);
        }
    });

    // Render Itinerary Cards Layout
    function renderItinerary(attractions) {
        itineraryContainer.innerHTML = '';
        
        if (!attractions || attractions.length === 0) {
            itineraryContainer.innerHTML = '<p class="empty-text">No attraction matches found for the budget/province filter.</p>';
            return;
        }

        attractions.forEach(day => {
            const dayBlock = document.createElement('div');
            dayBlock.className = 'day-block';
            
            // Day Header
            dayBlock.innerHTML = `
                <div class="day-header">
                    <i class="fa-solid fa-calendar-day"></i>
                    <span>Day ${day.day}</span>
                </div>
            `;

            // Morning Slot
            if (day.morning && day.morning.length > 0) {
                const morningSlot = document.createElement('div');
                morningSlot.className = 'time-slot';
                morningSlot.innerHTML = `
                    <div class="time-label"><i class="fa-solid fa-sun"></i> Morning Activity</div>
                    <div class="cards-grid">
                        ${day.morning.map(item => createCardMarkup(item)).join('')}
                    </div>
                `;
                dayBlock.appendChild(morningSlot);
            }

            // Evening Slot
            if (day.evening && day.evening.length > 0) {
                const eveningSlot = document.createElement('div');
                eveningSlot.className = 'time-slot';
                eveningSlot.innerHTML = `
                    <div class="time-label"><i class="fa-solid fa-moon"></i> Evening Activity</div>
                    <div class="cards-grid">
                        ${day.evening.map(item => createCardMarkup(item)).join('')}
                    </div>
                `;
                dayBlock.appendChild(eveningSlot);
            }

            itineraryContainer.appendChild(dayBlock);
        });
    }

    // Helper to create card markup
    function createCardMarkup(item) {
        // Fallback for image paths
        let imgUrl = item.image;
        if (!imgUrl) {
            imgUrl = 'downloads/noimage.jpg';
        }
        
        // Clean paths for browser rendering
        imgUrl = imgUrl.replace(/\\/g, '/');

        // Capitalize name/category
        const cleanName = item.name.replace(/_/g, ' ').toUpperCase();
        const cleanCat = item.category.replace(/_/g, ' ').toUpperCase();
        const ratingVal = item.rating ? parseFloat(item.rating).toFixed(1) : 'N/A';
        const priceVal = item.price ? `$${parseFloat(item.price).toFixed(2)}` : 'Free';

        return `
            <div class="recc-card">
                <div class="card-img-container">
                    <img src="/${imgUrl}" alt="${cleanName}" onerror="this.onerror=null;this.src='/downloads/noimage.jpg';">
                    <span class="badge">${ratingVal} ★</span>
                </div>
                <div class="card-info">
                    <h4>${cleanName}</h4>
                    <div class="card-meta">
                        <span class="card-category"><i class="fa-solid fa-tags"></i> ${cleanCat}</span>
                        <span class="price-tag">${priceVal}</span>
                    </div>
                </div>
            </div>
        `;
    }

    // Render Hotel Cards Layout
    function renderHotels(hotels) {
        hotelsContainer.innerHTML = '';

        if (!hotels || hotels.length === 0) {
            hotelsContainer.innerHTML = '<p class="empty-text">No hotel recommendations matches found for this province/amenities selection.</p>';
            return;
        }

        hotels.forEach(hotel => {
            const card = document.createElement('div');
            card.className = 'recc-card';

            let imgUrl = hotel.image || 'downloads/noimage.jpg';
            imgUrl = imgUrl.replace(/\\/g, '/');
            
            const hotelName = hotel.name.replace(/_/g, ' ');
            const priceVal = hotel.price ? `$${hotel.price}` : 'Unknown';
            const ratingVal = hotel.rating ? hotel.rating.toFixed(1) : 'N/A';

            card.innerHTML = `
                <div class="card-img-container">
                    <img src="/${imgUrl}" alt="${hotelName}" onerror="this.onerror=null;this.src='/downloads/noimage.jpg';">
                    <span class="badge">${hotel.experience}</span>
                </div>
                <div class="card-info">
                    <h4>${hotelName}</h4>
                    <div class="card-meta">
                        <span class="rating-badge"><i class="fa-solid fa-star"></i> ${ratingVal}</span>
                        <span class="price-tag">${priceVal} / Night</span>
                    </div>
                    <div class="card-address">
                        <i class="fa-solid fa-map-location-dot"></i>
                        <span>${hotel.address}</span>
                    </div>
                    <div class="card-amenities">
                        <strong>Amenities:</strong> ${hotel.amenities || 'Not listed'}
                    </div>
                </div>
            `;
            hotelsContainer.appendChild(card);
        });
    }

    // Helper functions
    function formatDate(date) {
        const yyyy = date.getFullYear();
        const mm = String(date.getMonth() + 1).padStart(2, '0');
        const dd = String(date.getDate()).padStart(2, '0');
        return `${yyyy}-${mm}-${dd}`;
    }

    function showError(message) {
        emptyState.style.display = 'flex';
        emptyState.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-orange);"></i>
            <h3>Recommendation Error</h3>
            <p>${message}</p>
            <button class="btn btn-primary" onclick="window.location.reload()" style="max-width: 200px; margin-top: 15px;">Retry</button>
        `;
    }
});
