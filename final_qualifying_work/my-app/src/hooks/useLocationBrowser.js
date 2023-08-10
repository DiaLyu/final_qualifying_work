import {useState, useEffect} from 'react';

const useLocationBrowser = () => {
    const [location, setLocation] = useState({
        loaded: false,
        coords: {lat: "", lng: ""}
    });

    const onSuccess = (location) => {
        setLocation({
            loaded: true,
            coords: {
                lat: location.coords.latitude,
                lng: location.coords.longitude
            }
        });
    };

    const onError = (error) => {
        setLocation({
            loaded: true, 
            error
        })
    };

    useEffect(() => {
        if (!("geolocation" in navigator)){
            onError({
                code: 0,
                message: "Геолокация"
            });
        }

        navigator.geolocation.getCurrentPosition(onSuccess, onError);
    }, []);

    return location;
}

export default useLocationBrowser;