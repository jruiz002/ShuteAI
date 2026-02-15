from shuteai.preprocesamiento.limpieza import (
    extraer_noticia,
    limpiar_texto,
    preparar_prompt,
    quitar_firmas,
)


def test_limpia_urls_emojis_y_espacios():
    t = limpiar_texto("¡URGENTE! 🚨🚨 mira   https://fake.com/x?a=1  esto\n\n www.otro.org")
    assert "http" not in t and "www" not in t and "🚨" not in t
    assert "  " not in t and t.startswith("¡URGENTE!")


def test_extrae_noticia_de_pregunta():
    assert (
        extraer_noticia("¿Es falsa esta noticia?: El agua será cortada 15 días")
        == "El agua será cortada 15 días"
    )
    assert extraer_noticia("Verifica esta noticia - Se prohíbe el pan") == "Se prohíbe el pan"


def test_sin_pregunta_no_altera_contenido():
    assert (
        extraer_noticia("El gobierno anunció nuevas medidas")
        == "El gobierno anunció nuevas medidas"
    )


def test_quita_firmas_y_medio():
    t = quitar_firmas(
        "(Reuters) - Hubo un accidente según El Heraldo Diario en la ciudad", "El Heraldo Diario"
    )
    assert "Reuters" not in t and "Heraldo" not in t


def test_advierte_texto_corto():
    r = preparar_prompt("hola")
    assert any("corto" in a for a in r.advertencias)


def test_detecta_idioma():
    assert (
        preparar_prompt(
            "El presidente anunció que el precio de la gasolina bajará en todo el país"
        ).idioma
        == "es"
    )
    assert (
        preparar_prompt(
            "The president announced that the price of gas will drop across the country"
        ).idioma
        == "en"
    )
